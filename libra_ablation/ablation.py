import os, argparse, json, math, csv, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from thop import profile

from models import Classifier, ImplicitReconstructor


def set_spectral_requires_grad(model, flag: bool):
    for m in model.modules():
        # LibraKAN
        if hasattr(m, "alpha"): m.alpha.requires_grad_(flag)
        if hasattr(m, "gate"):  m.gate.requires_grad_(flag)
        if hasattr(m, "freq") and isinstance(getattr(m, "freq"), torch.nn.Parameter):
            m.freq.requires_grad_(flag)
        # KAF 无需


def set_rho(model, value: float):
    for m in model.modules():
        if hasattr(m, "rho"):
            m.rho = float(value)


@torch.no_grad()
def sanitize_parameters(model):
    for m in model.modules():
        # 任何 NaN/Inf → 0
        if hasattr(m, "alpha"):
            m.alpha.data = torch.nan_to_num(m.alpha.data, nan=0.0, posinf=0.0, neginf=0.0)
            m.alpha.data.clamp_(-0.1, 0.1)  # 再夹一次
        if hasattr(m, "gate"):
            m.gate.data = torch.nan_to_num(m.gate.data, nan=0.0, posinf=0.0, neginf=0.0)
            m.gate.data.clamp_(0.0, 0.1)  # gate 非负小范围
        if hasattr(m, "freq") and isinstance(getattr(m, "freq"), torch.nn.Parameter):
            m.freq.data = torch.nan_to_num(m.freq.data, nan=0.0, posinf=0.0, neginf=0.0)
            m.freq.data.clamp_(-math.pi, math.pi)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def psnr(pred, target, eps=1e-12):
    mse = F.mse_loss(pred, target)
    return 10.0 * torch.log10(1.0 / (mse + eps))

def count_params_m(model): return sum(p.numel() for p in model.parameters())/1e6

@torch.no_grad()
def get_flops_g(model, example_inputs):
    try:
        macs, params = profile(model, inputs=example_inputs, verbose=False)
        return float(macs/1e9)
    except Exception:
        return -1.0

def make_coords(h, w, device):
    ys = torch.linspace(-1, 1, steps=h, device=device)
    xs = torch.linspace(-1, 1, steps=w, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1).view(-1, 2)

def clip_grad(model, max_norm=1.0):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

def run_cls(args):
    tfm = transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST if args.dataset=="mnist" else datasets.FashionMNIST
    train_set = ds(args.data_root, train=True, download=True, transform=tfm)
    test_set  = ds(args.data_root, train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)
    test_loader  = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)

    mkw = dict(
        Fmax=args.Fmax, rho=args.rho, spectral_scale=args.spectral_scale, es_beta=args.es_beta,
        tau_soft=args.tau, tau_learnable=args.tau_learnable, dropout=args.dropout,
        use_layernorm=not args.no_ln, use_cos_pair=not args.no_cos, spectral_dropout=args.spec_dropout,
        no_soft=args.wo_soft, uniform_freq=args.uniform, no_es=args.wo_es,
        p_sparse=args.p_sparse, gate_init=args.gate_init, alpha_init=args.alpha_init, max_freq_scale=args.max_freq_scale
    )
    model = Classifier(in_dim=28*28, width=args.width, depth=args.depth, num_classes=10,
                       mixer=args.mixer, **mkw).cuda().train()

    params_m = count_params_m(model)
    flops_g = get_flops_g(model, (torch.randn(1,1,28,28).cuda(),))

    spec_params, base_params = [], []
    for n, p in model.named_parameters():
        if any(k in n for k in ["core.alpha", "core.gate", "core.freq"]):
            spec_params.append(p)
        else:
            base_params.append(p)

    opt = torch.optim.Adam(
        [
            {"params": base_params, "lr": args.lr, "weight_decay": 1e-4},
            {"params": spec_params, "lr": args.lr * 0.1, "weight_decay": 1e-4},
        ],
        betas=(0.9, 0.999)
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def set_freq_trainable(flag: bool):
        for m in model.modules():
            if hasattr(m, "freq") and isinstance(m.freq, torch.nn.Parameter):
                m.freq.requires_grad_(flag)

    set_freq_trainable(False)

    set_spectral_requires_grad(model, False)
    rho_target = args.rho
    set_rho(model, 0.0)

    best = 0.0
    for ep in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"CLS Epoch {ep + 1}/{args.epochs}")
        for x, y in pbar:
            x = x.cuda(non_blocking=True);
            y = y.cuda(non_blocking=True)
            logits = model(x)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

            loss = F.cross_entropy(logits, y)
            reg_s = torch.tensor(0., device='cuda');
            reg_b = torch.tensor(0., device='cuda')
            for m in model.modules():
                if hasattr(m, "extra_losses"):
                    r = m.extra_losses(alpha_l1=args.alpha_sparse, beta_bal=args.alpha_balance, use_hspec=True)
                    reg_s += r["l_sparse"];
                    reg_b += r["l_balance"]
            loss = loss + reg_s + reg_b

            # 防御：非有限直接跳过这步，并做参数杀毒
            if not torch.isfinite(loss):
                sanitize_parameters(model)
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sanitize_parameters(model)

            pbar.set_postfix(loss=float(loss.item()))
        sched.step()

        # ---- 阶段切换逻辑 ----
        if ep == 0:
            # 转阶段 B：解冻频谱 + 低 rho
            set_spectral_requires_grad(model, True)
            set_rho(model, max(0.05, 0.25 * rho_target))
        elif ep >= 1:
            # 阶段 C：把 rho 线性拉到目标（最多 3 个 epoch 内到位）
            factor = min(1.0, (ep - 0 + 1) / 3.0)  # ep=1 -> 2/3; ep=2 -> 1.0
            set_rho(model, rho_target * factor)

        # ---- eval ----
        model.eval();
        correct = 0;
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.cuda();
                y = y.cuda()
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item();
                total += y.numel()
        acc = 100.0 * correct / total
        best = max(best, acc)
        model.train()

    def active(model, tau=1e-3):
        cnt=0
        for m in model.modules():
            if hasattr(m, "active_freq"): cnt+=m.active_freq(tau)
        return int(cnt)
    summary = dict(task="cls", dataset=args.dataset, mixer=args.mixer, width=args.width, depth=args.depth,
                   params_M=params_m, flops_G=flops_g, top1=best, active_freq=active(model, args.tau_active),
                   rho=args.rho, tau=args.tau, p_sparse=args.p_sparse)
    return model, summary

def run_imp(args):
    tfm = transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST if args.dataset=="mnist" else datasets.FashionMNIST
    test_set  = ds(args.data_root, train=False, download=True, transform=tfm)
    loader  = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)

    mkw = dict(
        Fmax=args.Fmax, rho=args.rho, spectral_scale=args.spectral_scale, es_beta=args.es_beta,
        tau_soft=args.tau, tau_learnable=args.tau_learnable, dropout=args.dropout,
        use_layernorm=not args.no_ln, use_cos_pair=not args.no_cos, spectral_dropout=args.spec_dropout,
        no_soft=args.wo_soft, uniform_freq=args.uniform, no_es=args.wo_es,
        p_sparse=args.p_sparse, gate_init=args.gate_init, alpha_init=args.alpha_init, max_freq_scale=args.max_freq_scale
    )
    model = ImplicitReconstructor(width=args.width, depth=args.depth, channels=1,
                                  mixer=args.mixer, **mkw).cuda().train()

    params_m = count_params_m(model)
    flops_g = get_flops_g(model, (torch.randn(28*28,2).cuda(),))

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.999))
    steps = args.epochs * 500
    x0, _ = next(iter(loader))
    x0 = x0.cuda()
    H, W = x0.shape[-2:]
    coords = make_coords(H,W, x0.device)
    target = x0.view(1,1,-1).permute(0,2,1).reshape(-1,1)

    best_psnr = 0.0
    for it in tqdm(range(1, steps+1), desc="IMP Train"):
        idx = torch.randint(0, H*W, (args.batch,), device=x0.device)
        y = model(coords[idx])
        loss = F.mse_loss(y, target[idx])
        reg_s, reg_b = torch.tensor(0.,device='cuda'), torch.tensor(0.,device='cuda')
        for m in model.modules():
            if hasattr(m, "extra_losses"):
                r = m.extra_losses(alpha_l1=args.alpha_sparse, beta_bal=args.alpha_balance, use_hspec=True)
                reg_s += r["l_sparse"]; reg_b += r["l_balance"]
        loss = loss + reg_s + reg_b
        if not torch.isfinite(loss):
            raise RuntimeError("Loss became non-finite.")
        opt.zero_grad(set_to_none=True); loss.backward(); clip_grad(model, args.grad_clip); opt.step()

        if it % 500 == 0:
            with torch.no_grad():
                pred = []
                BS = 1<<16
                for s in range(0, coords.size(0), BS):
                    pred.append(model(coords[s:s+BS]))
                pred = torch.cat(pred, dim=0).view(1,1,H,W)
                cur = psnr(pred, x0).item()
                best_psnr = max(best_psnr, cur)

    def active(model, tau=1e-3):
        cnt=0
        for m in model.modules():
            if hasattr(m, "active_freq"): cnt+=m.active_freq(tau)
        return int(cnt)
    summary = dict(task="imp", dataset=args.dataset, mixer=args.mixer, width=args.width, depth=args.depth,
                   params_M=params_m, flops_G=flops_g, psnr=best_psnr, active_freq=active(model, args.tau_active),
                   rho=args.rho, tau=args.tau, p_sparse=args.p_sparse)
    return model, summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, choices=["mnist","fmnist"], default="mnist")
    ap.add_argument("--task", type=str, choices=["cls","imp"], default="cls")
    ap.add_argument("--mixer", type=str, choices=["mlp","kaf","librakan"], default="librakan")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--Fmax", type=int, default=None)
    ap.add_argument("--rho", type=float, default=0.25)
    ap.add_argument("--spectral_scale", type=float, default=1.0)
    ap.add_argument("--es_beta", type=float, default=6.0)
    ap.add_argument("--tau", type=float, default=1e-3)
    ap.add_argument("--tau_learnable", default=False)
    ap.add_argument("--alpha_sparse", type=float, default=1e-4)
    ap.add_argument("--alpha_balance", type=float, default=1e-3)
    ap.add_argument("--p_sparse", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--spec_dropout", type=float, default=0.0)
    ap.add_argument("--no_ln", action="store_true")
    ap.add_argument("--no_cos", action="store_true")
    ap.add_argument("--wo_soft", action="store_true")
    ap.add_argument("--uniform", action="store_true")
    ap.add_argument("--wo_es", action="store_true")
    ap.add_argument("--tau_active", type=float, default=1e-3)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--gate_init", type=float, default=0.001)
    ap.add_argument("--alpha_init", type=float, default=0.001)
    ap.add_argument("--max_freq_scale", type=float, default=2.0)

    ap.add_argument("--out", type=str, default="runs/ablation")
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    set_seed(args.seed)

    if args.task == "cls":
        model, summ = run_cls(args)
    else:
        model, summ = run_imp(args)

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=2)

    csvp = os.path.join(os.path.dirname(args.out), "summary.csv")
    write_header = (not os.path.exists(csvp))
    cols = sorted(summ.keys())
    with open(csvp, "a", newline="") as f:
        w = csv.writer(f)
        if write_header: w.writerow(cols)
        w.writerow([summ[k] for k in cols])
    print("[DONE] Summary:", summ)

if __name__ == "__main__":
    main()
