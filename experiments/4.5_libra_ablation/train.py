import argparse, os, json, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import get_dataset
from mixers import make_mixer

def build_model(mixer, in_dim, num_classes, width=256, libra_kwargs=None):
    libra_kwargs = libra_kwargs or {}
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(in_dim, width)
            self.mixer = make_mixer(mixer, width, width, **libra_kwargs)
            self.head = nn.Linear(width, num_classes)
        def forward(self, x):
            x = x.view(x.size(0), -1).float()
            h = F.gelu(self.proj(x))
            h = self.mixer(h)
            return self.head(h)
    return Net()

def train_one_epoch(model, loader, opt, device):
    model.train()
    ce = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = ce(logits, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(1)
        total += y.size(0)
        correct += (pred == y).sum().item()
    return loss_sum/total, correct/total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(1)
        total += y.size(0)
        correct += (pred == y).sum().item()
    return loss_sum/total, correct/total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10","cifar100","mnist"])
    ap.add_argument("--mixer", type=str, default="librakan", choices=["mlp","kaf","kat","kan","librakan"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=str, default="results")
    # Libra knobs
    ap.add_argument("--F", type=int, default=256)
    ap.add_argument("--spectral_scale", type=float, default=1.0)
    ap.add_argument("--es_beta", type=float, default=6.0)
    ap.add_argument("--es_fmax", type=float, default=None)
    ap.add_argument("--nufft_dim", type=int, default=1, choices=[1,2])
    ap.add_argument("--lambda_init", type=float, default=0.01)
    ap.add_argument("--lambda_trainable", action="store_true")
    ap.add_argument("--p_trainable", action="store_true")
    ap.add_argument("--p_fixed", type=float, default=0.5)
    # Ablation toggles
    ap.add_argument("--disable_spectral", action="store_true")
    ap.add_argument("--oversamp", type=int, default=2)
    ap.add_argument("--grid_T", type=float, default=3.0)
    args = ap.parse_args()

    train_set, in_dim, num_classes = get_dataset(args.dataset, train=True)
    test_set, _, _ = get_dataset(args.dataset, train=False)
    train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)

    libra_kwargs = dict(
        F=args.F,
        spectral_scale=args.spectral_scale,
        es_beta=args.es_beta,
        es_fmax=args.es_fmax,
        nufft_dim=args.nufft_dim,
        lambda_init=args.lambda_init,
        lambda_trainable=args.lambda_trainable,
        p_fixed=args.p_fixed,
        p_trainable=args.p_trainable,
        base_activation="gelu",
        use_layernorm=False,
        dropout=0.0,
    )

    if args.disable_spectral and args.mixer == "librakan":
        libra_kwargs.update(F=0)  # minimal way to disable spectral branch

    device = torch.device(args.device)
    model = build_model(args.mixer, in_dim, num_classes, width=args.width, libra_kwargs=libra_kwargs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = 0.0
    hist = {"train_acc":[], "test_acc":[], "train_loss":[], "test_loss":[], "active_freq": []}
    for ep in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, device)
        te_loss, te_acc = evaluate(model, test_loader, device)
        hist["train_loss"].append(tr_loss); hist["test_loss"].append(te_loss)
        hist["train_acc"].append(tr_acc);  hist["test_acc"].append(te_acc)
        best = max(best, te_acc)
        # Active-Freq
        af = None
        try:
            libra = model.mixer.libra if hasattr(model.mixer, "libra") else None
            if libra is not None and getattr(libra, "last_abs_z", None) is not None:
                af = int((libra.last_abs_z > 1e-3).any(dim=0).sum().item())
        except Exception:
            af = None
        hist["active_freq"].append(af)

    os.makedirs(args.out, exist_ok=True)
    # Save per-run JSON with all details for Table 6
    details = {
        "dataset": args.dataset,
        "mixer": args.mixer,
        "epochs": args.epochs,
        "bs": args.bs,
        "width": args.width,
        "lr": args.lr,
        "F": args.F,
        "spectral_scale": args.spectral_scale,
        "es_beta": args.es_beta,
        "es_fmax": args.es_fmax,
        "nufft_dim": args.nufft_dim,
        "lambda_init": args.lambda_init,
        "lambda_trainable": args.lambda_trainable,
        "p_trainable": args.p_trainable,
        "p_fixed": args.p_fixed,
        "disable_spectral": args.disable_spectral,
        "best_acc": best,
        "last_active_freq": hist["active_freq"][-1] if hist["active_freq"] else None,
        "history": hist,
    }
    tag = f"{args.dataset}_{args.mixer}_F{args.F}_p{'learn' if args.p_trainable else args.p_fixed}_lam{'learn' if args.lambda_trainable else args.lambda_init}_dim{args.nufft_dim}"
    with open(os.path.join(args.out, f"ablation_{tag}.json"), "w") as f:
        json.dump(details, f, indent=2)

if __name__ == "__main__":
    main()
