import argparse, os, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from datasets import get_dataset
from mixers import make_mixer

def build_model(mixer, in_dim, num_classes, width=256, libra_kwargs=None):
    libra_kwargs = libra_kwargs or {}
    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(in_dim, width)
            self.block = make_mixer(mixer, width, width, **libra_kwargs)
            self.head = nn.Linear(width, num_classes)
        def forward(self, x):
            x = x.view(x.size(0), -1).float()
            h = F.gelu(self.proj(x))
            h = self.block(h)
            return self.head(h)
    return Classifier()

def train_one_epoch(model, loader, opt, device):
    model.train()
    ce = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in tqdm(loader, leave=False):
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
    ap.add_argument("--dataset", type=str, default="cifar10",
                    choices=["mnist","emnistb","emnistl","fmnist","kmnist","cifar10","cifar100","svhn"])
    ap.add_argument("--mixer", type=str, default="librakan", choices=["mlp","kan","kaf","kat","librakan"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # LibraKAN knobs (upgraded)
    ap.add_argument("--F", type=int, default=256)
    ap.add_argument("--spectral_scale", type=float, default=1.0)
    ap.add_argument("--es_beta", type=float, default=6.0)
    ap.add_argument("--es_fmax", type=float, default=None)
    ap.add_argument("--nufft_dim", type=int, default=1, choices=[1,2])
    ap.add_argument("--lambda_init", type=float, default=0.01)
    ap.add_argument("--lambda_trainable", action="store_true")
    ap.add_argument("--p_trainable", action="store_true")
    ap.add_argument("--p_fixed", type=float, default=0.5)

    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
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

    device = torch.device(args.device)
    model = build_model(args.mixer, in_dim, num_classes, width=args.width, libra_kwargs=libra_kwargs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    hist = {"train_acc":[], "test_acc":[], "train_loss":[], "test_loss":[]}
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, device)
        te_loss, te_acc = evaluate(model, test_loader, device)
        hist["train_loss"].append(tr_loss); hist["test_loss"].append(te_loss)
        hist["train_acc"].append(tr_acc);  hist["test_acc"].append(te_acc)
        best_acc = max(best_acc, te_acc)
        print(f"Epoch {epoch:03d}: train_acc={tr_acc*100:.2f} test_acc={te_acc*100:.2f}")

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(hist["train_acc"], label="train")
    plt.plot(hist["test_acc"], label="test")
    plt.xlabel("epoch"); plt.ylabel("accuracy")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out, f"acc_{args.dataset}_{args.mixer}.png"))

if __name__ == "__main__":
    main()
