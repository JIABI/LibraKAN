
import argparse, os, json
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision as tv
import torchvision.transforms as T
from mixers import make_mixer

class DeiTTinyLike(nn.Module):
    def __init__(self, num_classes, mixer_name, libra_kwargs, img=32, patch=4, dim=192, depth=8, heads=3, mlp_hidden=768):
        super().__init__()
        self.patch = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        n = (img//patch)*(img//patch)
        self.cls = nn.Parameter(torch.zeros(1,1,dim))
        self.pos = nn.Parameter(torch.zeros(1, n+1, dim))
        self.blocks = nn.ModuleList([self._block(dim, heads, mlp_hidden, mixer_name, libra_kwargs) for _ in range(depth)])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
    def _block(self, dim, heads, mlp_hidden, mixer_name, libra_kwargs):
        return nn.ModuleDict({
            "ln1": nn.LayerNorm(dim),
            "attn": nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True),
            "ln2": nn.LayerNorm(dim),
            "mlp": make_mixer(mixer_name, dim, mlp_hidden, **libra_kwargs)
        })
    def forward(self, x):
        x = self.patch(x).flatten(2).transpose(1,2)
        cls = self.cls.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos
        for b in self.blocks:
            h = b["ln1"](x); h,_ = b["attn"](h,h,h,need_weights=False); x = x + h
            h = b["ln2"](x); h = b["mlp"](h); x = x + h
        x = self.ln(x)[:,0]
        return self.head(x)

def get_cifar100(bs):
    tfm = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()])
    tfm_t = T.Compose([T.ToTensor()])
    train = tv.datasets.CIFAR100("./data", train=True, download=True, transform=tfm)
    test  = tv.datasets.CIFAR100("./data", train=False, download=True, transform=tfm_t)
    return (DataLoader(train, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True),
            DataLoader(test,  batch_size=bs, shuffle=False, num_workers=4, pin_memory=True))

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
    ap.add_argument("--epochs", type=int, default=200)
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

    train_loader, test_loader = get_cifar100(args.bs)
    device = torch.device(args.device)
    model = DeiTTinyLike(100, args.mixer, libra_kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = 0.0
    os.makedirs(args.out, exist_ok=True)
    for ep in range(1, args.epochs+1):
        train_one_epoch(model, train_loader, opt, device)
        vl, va = evaluate(model, test_loader, device)
        best = max(best, va)
        sched.step()
        print(f"[{ep:03d}] top1={va*100:.2f} best={best*100:.2f}")
    with open(os.path.join(args.out, f"deit_tiny_cifar100_{args.mixer}.json"), "w") as f:
        json.dump({"task":"deit_tiny_cifar100","mixer":args.mixer,"best_top1":best}, f, indent=2)

if __name__ == "__main__":
    main()
