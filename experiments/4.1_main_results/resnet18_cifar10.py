
import argparse, os, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision as tv
import torchvision.transforms as T
from mixers import make_mixer

def get_cifar10(bs):
    tfm = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()])
    tfm_t = T.Compose([T.ToTensor()])
    train = tv.datasets.CIFAR10("./data", train=True, download=True, transform=tfm)
    test  = tv.datasets.CIFAR10("./data", train=False, download=True, transform=tfm_t)
    return (DataLoader(train, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True),
            DataLoader(test,  batch_size=bs, shuffle=False, num_workers=4, pin_memory=True))

class ResNet18HeadMixer(nn.Module):
    # Replace only the pre-classifier head by a bottleneck + mixer
    def __init__(self, backbone, feat_dim, num_classes, mixer_name, hidden, libra_kwargs):
        super().__init__()
        self.backbone = backbone
        self.proj = nn.Linear(feat_dim, hidden)
        self.mixer = make_mixer(mixer_name, hidden, hidden, **libra_kwargs)
        self.cls = nn.Linear(hidden, num_classes)
    def forward(self, x):
        x = self.backbone(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        h = F.gelu(self.proj(x))
        h = self.mixer(h)
        return self.cls(h)

def build_model(mixer_name, hidden, libra_kwargs, num_classes=10):
    backbone = tv.models.resnet18(weights=None, num_classes=None)
    feat_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return ResNet18HeadMixer(backbone, feat_dim, num_classes, mixer_name, hidden, libra_kwargs)

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
    ap.add_argument("--mixer", type=str, default="mlp", choices=["mlp","kaf","kat","fan","kan","librakan"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # Libra options
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

    train_loader, test_loader = get_cifar10(args.bs)
    device = torch.device(args.device)
    model = build_model(args.mixer, args.hidden, libra_kwargs).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = 0.0
    os.makedirs(args.out, exist_ok=True)
    for ep in range(1, args.epochs+1):
        train_one_epoch(model, train_loader, opt, device)
        vl, va = evaluate(model, test_loader, device)
        best = max(best, va)
        sched.step()
        print(f"[{ep:03d}] acc={va*100:.2f} best={best*100:.2f}")
    # save metrics
    out_js = {"task":"resnet18_cifar10","mixer":args.mixer,"best_top1":best}
    with open(os.path.join(args.out, f"resnet18_cifar10_{args.mixer}.json"), "w") as f:
        json.dump(out_js, f, indent=2)

if __name__ == "__main__":
    main()
