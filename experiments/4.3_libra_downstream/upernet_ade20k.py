import argparse, os, json, torch, torch.nn as nn, torch.nn.functional as F
import torchvision as tv
from torch.utils.data import DataLoader
from mixers import make_mixer
from PIL import Image
import numpy as np

class ADE20KDataset(torch.utils.data.Dataset):
    def __init__(self, root, split="train", img_size=512):
        self.imgs = os.path.join(root, "images", split)
        self.anns = os.path.join(root, "annotations", split)
        self.files = sorted(os.listdir(self.imgs))
        self.img_size = img_size
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        name = self.files[idx]
        img = Image.open(os.path.join(self.imgs, name)).convert("RGB")
        ann = Image.open(os.path.join(self.anns, name.replace(".jpg",".png")))
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        ann = ann.resize((self.img_size, self.img_size), Image.NEAREST)
        img = torch.from_numpy(np.array(img)).permute(2,0,1).float()/255.0
        ann = torch.from_numpy(np.array(ann)).long()
        return img, ann

class UPerHeadLite(nn.Module):
    def __init__(self, in_channels_list, num_classes, mixer_name, hidden, libra_kwargs):
        super().__init__()
        self.laterals = nn.ModuleList([nn.Conv2d(c, 256, kernel_size=1) for c in in_channels_list])
        self.ppm = nn.ModuleList([nn.AdaptiveAvgPool2d(s) for s in (1,2,3,6)])
        self.fuse = nn.Conv2d(256*(1+len(self.ppm)), 256, kernel_size=3, padding=1)
        self.pre = nn.Conv2d(256, hidden, kernel_size=1)
        self.mixer = make_mixer(mixer_name, hidden, hidden, **libra_kwargs)
        self.post = nn.Conv2d(hidden, num_classes, kernel_size=1)
    def forward(self, feats):
        laterals = [l(f) for l,f in zip(self.laterals, feats)]
        for i in range(2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(laterals[i+1], size=laterals[i].shape[-2:], mode="bilinear", align_corners=False)
        x = laterals[0]
        xs = [x]
        for pool in self.ppm:
            p = pool(x)
            p = F.interpolate(p, size=x.shape[-2:], mode="bilinear", align_corners=False)
            xs.append(p)
        x = torch.cat(xs, dim=1)
        x = self.fuse(x).relu_()
        b,c,h,w = x.shape
        y = self.pre(x).permute(0,2,3,1).reshape(-1, self.pre.out_channels)
        y = self.mixer(y)
        y = y.reshape(b,h,w,-1).permute(0,3,1,2)
        out = self.post(y)
        return out

class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        m = tv.models.resnet50(weights=None)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4
    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c2, c3, c4, c5]

def train_one_epoch(model, loader, opt, device, num_classes):
    model.train()
    loss_sum = 0.0
    for img, ann in loader:
        img, ann = img.to(device), ann.to(device)
        logits = model(img)
        loss = F.cross_entropy(logits, ann, ignore_index=255)
        opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += loss.item() * img.size(0)
    return loss_sum/len(loader.dataset)

@torch.no_grad()
def evaluate_miou(model, loader, device, num_classes):
    model.eval()
    import numpy as np
    hist = np.zeros((num_classes, num_classes), dtype=np.int64)
    for img, ann in loader:
        img = img.to(device)
        pred = model(img).argmax(1).cpu().numpy()
        gt = ann.numpy()
        for p, g in zip(pred, gt):
            m = (g>=0) & (g<num_classes)
            hist += np.bincount(g[m]*num_classes + p[m], minlength=num_classes**2).reshape(num_classes, num_classes)
    iu = np.diag(hist)/(hist.sum(1)+hist.sum(0)-np.diag(hist)+1e-6)
    return float(np.nanmean(iu))

class Net(nn.Module):
    def __init__(self, bb, hd): super().__init__(); self.bb=bb; self.hd=hd
    def forward(self, x): feats = self.bb(x); return self.hd(feats)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ade20k_root", type=str, required=True)
    ap.add_argument("--mixer", type=str, default="mlp", choices=["mlp","kaf","kat","fan","kan","librakan"])
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_classes", type=int, default=150)
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=str, default="results")
    # Libra
    ap.add_argument("--F", type=int, default=256)
    ap.add_argument("--spectral_scale", type=float, default=1.0)
    ap.add_argument("--es_beta", type=float, default=6.0)
    ap.add_argument("--es_fmax", type=float, default=None)
    ap.add_argument("--nufft_dim", type=int, default=1)
    ap.add_argument("--lambda_init", type=float, default=0.01)
    ap.add_argument("--lambda_trainable", action="store_true")
    ap.add_argument("--p_trainable", action="store_true")
    ap.add_argument("--p_fixed", type=float, default=0.5)
    args = ap.parse_args()

    libra_kwargs = dict(F=args.F, spectral_scale=args.spectral_scale, es_beta=args.es_beta, es_fmax=args.es_fmax,
                        nufft_dim=args.nufft_dim, lambda_init=args.lambda_init, lambda_trainable=args.lambda_trainable,
                        p_fixed=args.p_fixed, p_trainable=args.p_trainable, base_activation="gelu",
                        use_layernorm=False, dropout=0.0)

    device = torch.device(args.device)
    train_set = ADE20KDataset(args.ade20k_root, "train", args.img_size)
    val_set   = ADE20KDataset(args.ade20k_root, "val", args.img_size)
    train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)

    backbone = ResNetBackbone()
    head = UPerHeadLite([256,512,1024,2048], args.num_classes, args.mixer, args.hidden, libra_kwargs)
    model = Net(backbone, head).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    best = 0.0
    for ep in range(1, args.epochs+1):
        tr = train_one_epoch(model, train_loader, opt, device, args.num_classes)
        miou = evaluate_miou(model, val_loader, device, args.num_classes)
        best = max(best, miou)
        print(f"[{ep:03d}] miou={miou:.4f} best={best:.4f}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"upernet_ade20k_{args.mixer}.json"), "w") as f:
        json.dump({"task":"ade20k_upernet","mixer":args.mixer,"miou":float(best)}, f, indent=2)

if __name__ == "__main__":
    main()
