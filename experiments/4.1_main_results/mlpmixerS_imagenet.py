
import argparse, os, json, math
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision as tv
import torchvision.transforms as T
from mixers import make_mixer

class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, patch=16, emb=512, img=224):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, emb, kernel_size=patch, stride=patch)
        self.n = (img//patch)*(img//patch)
    def forward(self, x):
        x = self.proj(x)   # (B,emb,H',W')
        x = x.flatten(2).transpose(1,2)  # (B,N,emb)
        return x

class MixerBlock(nn.Module):
    def __init__(self, n_tokens, emb_dim, token_mlp_dim, channel_mlp_dim, mixer_name, libra_kwargs):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        # token mixing: MLP over tokens (transpose) – keep baseline MLP here for fairness
        self.token_mlp = nn.Sequential(nn.Linear(n_tokens, token_mlp_dim), nn.GELU(), nn.Linear(token_mlp_dim, n_tokens))
        self.ln2 = nn.LayerNorm(emb_dim)
        # channel mixing: replaceable by mixer
        self.channel_mixer = make_mixer(mixer_name, emb_dim, channel_mlp_dim, **libra_kwargs)
    def forward(self, x):
        y = self.ln1(x)
        y = y.transpose(1,2)
        y = self.token_mlp(y)
        y = y.transpose(1,2)
        x = x + y
        y = self.ln2(x)
        y = self.channel_mixer(y)
        return x + y

class MLPMixerSmall(nn.Module):
    # MLP-Mixer/S config: (img 224, patch 16, emb 512, depth 8, token_mlp 256, channel_mlp 2048)
    def __init__(self, num_classes, mixer_name, libra_kwargs):
        super().__init__()
        self.pe = PatchEmbed(3, 16, 512, 224)
        n_tokens = (224//16)*(224//16)
        blocks = []
        for _ in range(8):
            blocks.append(MixerBlock(n_tokens, 512, token_mlp_dim=256, channel_mlp_dim=2048, mixer_name=mixer_name, libra_kwargs=libra_kwargs))
        self.blocks = nn.ModuleList(blocks)
        self.ln = nn.LayerNorm(512)
        self.head = nn.Linear(512, num_classes)
    def forward(self, x):
        x = self.pe(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x).mean(dim=1)
        return self.head(x)

def build_imagenet(root, bs, img_size=224):
    train_t = T.Compose([T.RandomResizedCrop(img_size), T.RandomHorizontalFlip(), T.ToTensor()])
    val_t   = T.Compose([T.Resize(256), T.CenterCrop(img_size), T.ToTensor()])
    train = tv.datasets.ImageFolder(os.path.join(root, "train"), transform=train_t)
    val   = tv.datasets.ImageFolder(os.path.join(root, "val"), transform=val_t)
    return (DataLoader(train, batch_size=bs, shuffle=True, num_workers=8, pin_memory=True),
            DataLoader(val,   batch_size=bs, shuffle=False, num_workers=8, pin_memory=True))

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    tot, corr, lsum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x); loss = ce(logits, y)
        lsum += loss.item() * x.size(0)
        corr += (logits.argmax(1) == y).sum().item()
        tot += y.size(0)
    return lsum/tot, corr/tot

def train_one_epoch(model, loader, opt, device):
    model.train()
    ce = nn.CrossEntropyLoss()
    tot, corr, lsum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x); loss = ce(logits, y)
        loss.backward(); opt.step()
        lsum += loss.item() * x.size(0)
        corr += (logits.argmax(1) == y).sum().item()
        tot += y.size(0)
    return lsum/tot, corr/tot

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagenet_root", type=str, required=True, help="Path to ImageNet-1k root with subfolders train/ and val/")
    ap.add_argument("--mixer", type=str, default="mlp", choices=["mlp","kaf","kat","fan","kan","librakan"])
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
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
    ap.add_argument("--out", type=str, default="results")
    args = ap.parse_args()

    libra_kwargs = dict(F=args.F, spectral_scale=args.spectral_scale, es_beta=args.es_beta, es_fmax=args.es_fmax,
                        nufft_dim=args.nufft_dim, lambda_init=args.lambda_init, lambda_trainable=args.lambda_trainable,
                        p_fixed=args.p_fixed, p_trainable=args.p_trainable, base_activation="gelu",
                        use_layernorm=False, dropout=0.0)

    train_loader, val_loader = build_imagenet(args.imagenet_root, args.bs)
    device = torch.device(args.device)
    model = MLPMixerSmall(1000, args.mixer, libra_kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = 0.0
    os.makedirs(args.out, exist_ok=True)
    for ep in range(1, args.epochs+1):
        train_one_epoch(model, train_loader, opt, device)
        vl, va = evaluate(model, val_loader, device)
        best = max(best, va)
        sched.step()
        print(f"[{ep:03d}] top1={va*100:.2f} best={best*100:.2f}")
    with open(os.path.join(args.out, f"mlpmixerS_imagenet_{args.mixer}.json"), "w") as f:
        json.dump({"task":"mlpmixerS_imagenet","mixer":args.mixer,"best_top1":best}, f, indent=2)

if __name__ == "__main__":
    main()
