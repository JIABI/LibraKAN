#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sine extrapolation using YOUR Blocks:
{MLPBlock, KAFBlockExt/RFFActivation, KANPyBlock, GPKANBlockExt, LibraKANBlockExt}

- Deterministic seeding (--seed)
- Fixed training set (no per-step re-sampling)
- Summary PDF written via PdfPages (no plt.imread on PDFs)

Example:
python sin_freq.py --mixer librakan --width 128 --layers 3 --freq 192 --shrink 1e-3 --l1_spec 1e-4 --steps 8000 --seed 3407
"""

import os, random, math, argparse, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from kaf_act import RFFActivation
# ----------------------------
# Seeding & CuDNN determinism
# ----------------------------
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 更强一致性（可能稍慢）
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------
# Try importing your modules (prefer your project structure)
# ----------------------------
def try_imports():
    mods = {}
    try:
        from models.MLP import MLPBlock as _MLPBlock
        mods["MLPBlock"] = _MLPBlock
    except Exception:
        mods["MLPBlock"] = None
    try:
        from models.KAF import KAFBlockExt as _KAFBlockExt
        mods["KAFBlockExt"] = _KAFBlockExt
    except Exception:
        mods["KAFBlockExt"] = None
    try:
        from kaf_act import RFFActivation as _RFFActivation
        mods["RFFActivation"] = _RFFActivation
    except Exception:
        mods["RFFActivation"] = None
    try:
        from models.KAN import KANPyBlock as _KANPyBlock
        mods["KANPyBlock"] = _KANPyBlock
    except Exception:
        mods["KANPyBlock"] = None
    try:
        from models.GPKAN import GPKANBlockExt as _GPKANBlockExt
        mods["GPKANBlockExt"] = _GPKANBlockExt
    except Exception:
        mods["GPKANBlockExt"] = None
    try:
        from models.LibraKAN import LibraKANBlockExt as _LibraKANBlockExt
        mods["LibraKANBlockExt"] = _LibraKANBlockExt
    except Exception:
        mods["LibraKANBlockExt"] = None
    return mods

mods = try_imports()


# ----------------------------
# Fallback building blocks
# ----------------------------
class FallbackMLPBlock(nn.Module):
    def __init__(self, dim_in, dim_out, activation="gelu", use_layernorm=False, dropout=0.0):
        super().__init__()
        act = nn.GELU() if activation == "gelu" else nn.SiLU(inplace=True)
        layers = [nn.Linear(dim_in, dim_out), act]
        if use_layernorm:
            layers.append(nn.LayerNorm(dim_out))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)


class FallbackKAFBlock(nn.Module):
    """Tiny surrogate: concat sin/cos features then project."""
    def __init__(self, dim_in, dim_out, num_grids=16, base_activation="gelu", dropout=0.0):
        super().__init__()
        Fmax = max(4, num_grids * 2)
        self.freq = nn.Parameter(torch.linspace(1.0, float(Fmax), Fmax), requires_grad=False)
        self.fc = nn.Linear(dim_in + 2 * Fmax, dim_out)
        self.act = nn.GELU() if base_activation == "gelu" else nn.SiLU(inplace=True)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    def forward(self, x):
        coord = x[:,:1]
        wx = coord*self.freq[None,:]
        #wx = x @ self.freq[None, :].t()  # (B,F)
        feats = torch.cat([x, torch.sin(wx), torch.cos(wx)], dim=-1)
        return self.drop(self.act(self.fc(feats)))

class FallbackLibraBlock(nn.Module):
    """Local Linear-GELU + simple learnable spectral with soft-shrink + fusion."""
    def __init__(self, dim_in, dim_out, F=64, lambda_init=1e-2, base_activation="gelu", dropout=0.0):
        super().__init__()
        self.local = nn.Sequential(nn.Linear(dim_in, dim_out),
                                   nn.GELU() if base_activation=="gelu" else nn.SiLU())
        self.omegas = nn.Parameter(torch.linspace(1.0, float(F), F))
        self.a_sin = nn.Parameter(0.01 * torch.randn(F))
        self.a_cos = nn.Parameter(0.01 * torch.randn(F))
        self.shrink = nn.Softshrink(lambd=lambda_init)
        self.out = nn.Linear(dim_out, dim_out, bias=True)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._last_active_ratio = torch.tensor(0.0)
    def forward(self, x):
        coord = x[:, :1]                                # (B,1)
        wx = coord * self.omegas[None, :]               # (B,F)
        spec = self.shrink(torch.sin(wx) * self.a_sin + torch.cos(wx) * self.a_cos)  # (B,F)
        with torch.no_grad():
            amp = torch.sqrt(self.a_sin ** 2 + self.a_cos ** 2)
            self._last_active_ratio = (amp.abs() > 1e-3).float().mean()
        y = self.local(x) + spec.sum(dim=-1, keepdim=True)  # fuse
        return self.drop(self.out(y))
    def stats(self):
        return {"active_ratio": float(self._last_active_ratio.item())}


# ----------------------------
# KAF adapter (prefer official RFFActivation)
# ----------------------------
class KAFBlockAdapter(nn.Module):
    """
    Closer-to-paper KAF block:
      Linear(dim_in->hidden) -> RFFActivation -> Linear(hidden->dim_out)
    默认关闭 LayerNorm/Dropout；设置 num_grids & activation_expectation。
    """

    def __init__(self, dim_in, dim_out, hidden=None, num_grids=16, activation_expectation=1.64):
        super().__init__()
        h = hidden or max(dim_in, dim_out)  # 你也可以直接用 dim_out 以严格对齐参数规模
        self.fc1 = nn.Linear(dim_in, h, bias=True)
        self.kaf = RFFActivation(
            num_grids=num_grids,
            activation_expectation=activation_expectation,
            use_layernorm=False,
            dropout=0.0,
            base_activation=F.gelu  # 他们示例里也常用 GELU；你保持和其它 mixer 一致
        )
        self.fc2 = nn.Linear(h, dim_out, bias=True)

    def forward(self, x):
        h = self.fc1(x)
        h = self.kaf(h)  # 按通道的 Fourier 激活
        y = self.fc2(h)
        return y


# ----------------------------
# Libra adapter (prefer your LibraKANBlockExt)
# ----------------------------
class LibraBlockAdapter(nn.Module):
    def __init__(self, dim_in, dim_out, F=64, lambda_init=1e-2, l1_alpha=0.0, base_activation="gelu"):
        super().__init__()
        if mods["LibraKANBlockExt"] is not None:
            self.block = mods["LibraKANBlockExt"](
                dim_in=dim_in, dim_out=dim_out, F=F, spectral_scale=1.0,
                es_beta=8.0, es_fmax=None, lambda_init=lambda_init,
                lambda_trainable=True, l1_alpha=l1_alpha,
                base_activation=base_activation, use_layernorm=False, dropout=0.0
            )
        else:
            self.block = FallbackLibraBlock(dim_in, dim_out, F=F, lambda_init=lambda_init, base_activation=base_activation)
    def forward(self, x): return self.block(x)
    @property
    def sparsity_loss(self):
        # if block exposes sparsity_loss, use it; else 0
        sl = getattr(self.block, "sparsity_loss", None)
        if isinstance(sl, torch.Tensor):
            return sl
        return torch.zeros((), device=x.device) if hasattr(self, "device") else torch.tensor(0.0)
    def stats(self):
        return self.block.stats() if hasattr(self.block, "stats") else {}


# ----------------------------
# Build a single block by mixer name
# ----------------------------
def build_block(mixer, dim_in, dim_out, args):
    name = mixer.lower()
    if name == "mlp":
        if mods['MLPBlock'] is not None:
            return mods['MLPBlock'](dim_in=dim_in, dim_out=dim_out, activation="gelu", use_layernorm=False, dropout=0.0)
        return FallbackMLPBlock(dim_in, dim_out)
    elif name == "kaf":
        return KAFBlockAdapter(dim_in, dim_out, hidden=dim_out, num_grids=args.num_grids, activation_expectation=1.64)
    elif name == "kan":
        if mods['KANPyBlock'] is not None:
            return mods['KANPyBlock'](in_features=dim_in, out_features=dim_out, grid=5, degree=3, x_range=1.0)
        return FallbackMLPBlock(dim_in, dim_out)
    elif name == "gpkan":
        if mods['GPKANBlockExt'] is not None:
            return mods['GPKANBlockExt'](dim_in=dim_in, dim_out=dim_out, mlp_ratio=4.0, drop=0.0, act_init="gelu")
        return FallbackMLPBlock(dim_in, dim_out)
    elif name == "librakan":
        return LibraBlockAdapter(dim_in, dim_out, F=args.freq, lambda_init=args.shrink, l1_alpha=args.l1_spec, base_activation="gelu")
    else:
        raise ValueError(f"Unknown mixer: {mixer}")


# ----------------------------
# Stacked regressor
# ----------------------------
class SineRegressor(nn.Module):
    def __init__(self, mixer="librakan", width=128, layers=3, args=None):
        super().__init__()
        dims = [1] + [width] * (layers - 1) + [1] if layers >= 1 else [1, 1]
        blocks = []
        for i in range(len(dims) - 1):
            blocks.append(build_block(mixer, dims[i], dims[i + 1], args))
        self.backbone = nn.Sequential(*blocks)
    def forward(self, x):  # x: (B,1)
        return self.backbone(x)
    def active_freq_ratio(self):
        tot, cnt = 0.0, 0
        for m in self.backbone.modules():
            if hasattr(m, "stats"):
                s = m.stats()
                if isinstance(s, dict) and "active_ratio" in s:
                    tot += float(s["active_ratio"]); cnt += 1
        return None if cnt == 0 else tot / cnt
    def extra_sparsity_loss(self, device):
        loss = torch.zeros((), device=device)
        for m in self.backbone.modules():
            if hasattr(m, "sparsity_loss") and isinstance(m.sparsity_loss, torch.Tensor):
                loss = loss + m.sparsity_loss.to(device)
        return loss


# ----------------------------
# Data & helpers
# ----------------------------
def make_fixed_trainset(n_total=4096, low=-math.pi/2, high=math.pi/2, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.uniform(low, high, size=(n_total, 1)).astype("float32")
    y = np.sin(x).astype("float32")
    return torch.from_numpy(x), torch.from_numpy(y)

def grid_eval(xmin=-3*math.pi, xmax=3*math.pi, n=4000):
    x = torch.linspace(xmin, xmax, n).unsqueeze(-1)
    y = torch.sin(x)
    return x, y

def in_out_masks(x):
    in_mask = (x[:, 0] >= -math.pi/2) & (x[:, 0] <= math.pi/2)
    out_mask = ~in_mask
    return in_mask, out_mask

def rmse(a, b): return torch.sqrt(F.mse_loss(a, b))

def train(model, steps=8000, batch=1024, lr=1e-3, device="cpu", weight_decay=0.0,
          train_x=None, train_y=None, seed=42):
    assert train_x is not None and train_y is not None
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # deterministic batch sampler
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    N = train_x.shape[0]
    tx = train_x.to(device)
    ty = train_y.to(device)

    for _ in range(steps):
        idx = torch.randint(0, N, (batch,), generator=g, device=device)
        x = tx.index_select(0, idx)
        y = ty.index_select(0, idx)
        pred = model(x)
        loss = F.mse_loss(pred, y) + model.extra_sparsity_loss(device)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model

def evaluate(model, device="cpu"):
    model.eval()
    xs, ys = grid_eval()
    xs, ys = xs.to(device), ys.to(device)
    with torch.no_grad():
        yp = model(xs)
    in_mask, out_mask = in_out_masks(xs)
    r_in = rmse(yp[in_mask], ys[in_mask]).item()
    r_out = rmse(yp[out_mask], ys[out_mask]).item()
    return xs.cpu(), ys.cpu(), yp.cpu(), r_in, r_out

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------
# Plotting
# ----------------------------
def plot(xs, ys, yp, save="sine_extrap.pdf"):
    import matplotlib.pyplot as plt
    xs = xs.squeeze().numpy(); ys = ys.squeeze().numpy(); yp = yp.squeeze().numpy()
    plt.figure(figsize=(9, 3.2))
    plt.axvspan(-3*math.pi, 3*math.pi, color="#CFE9FF", alpha=0.35, label="evaluation span")
    plt.axvspan(-math.pi/2, math.pi/2, color="#CFF7CF", alpha=0.45, label="training interval")
    plt.plot(xs, ys, lw=2, color="black", label="ground truth")
    plt.plot(xs, yp, lw=2, ls="--", label="prediction")
    plt.xlim([-3*math.pi, 3*math.pi]); plt.xlabel("x"); plt.ylabel("y")
    plt.legend(frameon=False, loc="upper right")
    plt.tight_layout(); plt.savefig(save, dpi=200)


def run_one(mixer, args, train_x, train_y):
    set_seed(args.seed)
    model = SineRegressor(mixer=mixer, width=args.width, layers=args.layers, args=args)
    model = train(model, steps=args.steps, batch=args.batch, lr=args.lr, device=args.device,
                  weight_decay=args.wd, train_x=train_x, train_y=train_y, seed=args.seed)
    xs, ys, yp, r_in, r_out = evaluate(model, device=args.device)
    act = model.active_freq_ratio()
    return {"mixer": mixer, "xs": xs, "ys": ys, "yp": yp, "rmse_in": r_in, "rmse_out": r_out, "act": act}


# -------- Option A: write figures directly into a combined PDF (no imread) --------
def plot_summary(results, save="comparison_summary.pdf"):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    xs = results[0]["xs"].squeeze().numpy()
    ys = results[0]["ys"].squeeze().numpy()
    mixers = [r["mixer"] for r in results]
    rmse_out = [r["rmse_out"] for r in results]
    acts = [(-1 if r["act"] is None else r["act"]) for r in results]

    with PdfPages(save) as pdf:
        # 1) overlay
        plt.figure(figsize=(11, 5))
        plt.axvspan(-3 * math.pi, 3 * math.pi, color="#CFE9FF", alpha=0.35)
        plt.axvspan(-math.pi / 2, math.pi / 2, color="#CFF7CF", alpha=0.45)
        plt.plot(xs, ys, lw=2, color="black", label="ground truth")
        for r in results:
            plt.plot(xs, r["yp"].squeeze().numpy(), lw=1.8, ls="--",
                     label=f'{r["mixer"]} ({r["rmse_out"]:.3e})')
        plt.xlim([-3 * math.pi, 3 * math.pi]);
        plt.xlabel("x");
        plt.ylabel("y")
        plt.legend(frameon=False, ncol=2, fontsize=12)
        plt.tight_layout()
        pdf.savefig(bbox_inches="tight", pad_inches=0.1)
        plt.close()

    print(f"Saved summary to {save}")


# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixer", type=str, default="librakan",
                    choices=["mlp", "kaf", "kan", "gpkan", "librakan"])
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3, help="number of blocks in the stack")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=3407)
    # KAF-like
    ap.add_argument("--num_grids", type=int, default=16)
    # LibraKAN-like
    ap.add_argument("--freq", type=int, default=64)
    ap.add_argument("--shrink", type=float, default=1e-2)
    ap.add_argument("--l1_spec", type=float, default=1e-4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--do_summary", action="store_true",
                    help="run a fixed set of mixers and produce comparison_summary.pdf")
    args = ap.parse_args()

    # deterministic
    set_seed(args.seed)

    # fixed trainset
    train_x, train_y = make_fixed_trainset(seed=args.seed)

    # single mixer run
    model = SineRegressor(mixer=args.mixer, width=args.width, layers=args.layers, args=args)
    n_params = count_params(model)
    print(f"[{args.mixer}] params={n_params/1e3:.1f}K  width={args.width}  layers={args.layers}")

    model = train(model, steps=args.steps, batch=args.batch, lr=args.lr, device=args.device,
                  weight_decay=args.wd, train_x=train_x, train_y=train_y, seed=args.seed)
    xs, ys, yp, r_in, r_out = evaluate(model, device=args.device)

    act = model.active_freq_ratio()
    if act is None:
        print(f"[{args.mixer}] RMSE(in)={r_in:.3e}  RMSE(out)={r_out:.3e}  Active-Freq=-")
    else:
        print(f"[{args.mixer}] RMSE(in)={r_in:.3e}  RMSE(out)={r_out:.3e}  Active-Ratio≈{act:.2f}")

    out = args.out or f"sine_extrap_{args.mixer}.pdf"
    plot(xs, ys, yp, save=out)
    print(f"Saved to {out}")

    # optional summary over multiple mixers
    if args.do_summary:
        mix_list = ["mlp", "kan", "kaf", "librakan"]  # add "gpkan" if your import works well
        results = [run_one(m, args, train_x, train_y) for m in mix_list]
        plot_summary(results, save="comparison_summary.pdf")


if __name__ == "__main__":
    main()
