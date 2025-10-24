
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, torch, matplotlib.pyplot as plt
from train_utils import set_seed, make_data, train_model
from models.librakan import MLP, LibraKAN1D
from models.kan import KAN1D
from models.kaf import KAF

# ============================================================
# 🎨 统一颜色方案
# ============================================================
PALETTE = {
    "target":   {"color": "#BDBDBD", "lw": 2.2, "alpha": 0.9, "z": 1},   # 浅灰
    "LibraKAN": {"color": "#0077FF", "lw": 2.4, "alpha": 1.0, "z": 5},   # 亮蓝
    "KAF":      {"color": "#2CA02C", "lw": 2.0, "alpha": 1.0, "z": 4},   # 绿色
    "MLP":      {"color": "#FF69B4", "lw": 2.0, "alpha": 1.0, "z": 3},   # 粉色
    "KAN":      {"color": "#FF7F0E", "lw": 2.0, "alpha": 1.0, "z": 2},   # 橙色
}

def _style_for(label: str):
    """根据曲线名称匹配颜色样式"""
    key = "target"
    u = label.lower()
    if "librakan" in u:
        key = "LibraKAN"
    elif "kaf" in u:
        key = "KAF"
    elif "mlp" in u:
        key = "MLP"
    elif "kan" in u:
        key = "KAN"
    st = PALETTE[key].copy()
    st["label"] = label
    return st

def plot_with_style(ax, x, y, label):
    """统一绘制接口"""
    st = _style_for(label)
    ax.plot(x, y, label=st["label"], color=st["color"],
            linewidth=st["lw"], alpha=st["alpha"], zorder=st["z"])

# ============================================================
# 🔧 模型运行
# ============================================================
def run_trio(fn, trio, device, epochs, amp):
    x_tr, y_tr, x_ev, y_ev = make_data(
        n_train=512, n_eval=2000, fn=fn,
        train_range=(-1.0,1.0), eval_range=(-10.0,10.0),
        device=device
    )
    preds = []
    if trio == "relu":
        mlp = MLP(act="relu"); kan = KAN1D(num_knots=256, hidden=None, use_layernorm=False, train_xmin=-5.0, train_xmax=5.0, act="None")
        libra = LibraKAN1D(spectral_dim=64, act="relu", k_active=16, fmax=4.0)
        models = [("MLP(ReLU)", mlp), ("KAN", kan), ("LibraKAN(ReLU)", libra)]
    else:
        mlp = MLP(act="gelu"); kaf = KAF([1,64,64,1], num_grids=16)
        libra = LibraKAN1D(spectral_dim=64, act="gelu", k_active=16, fmax=4.0)
        models = [("MLP(GELU)", mlp), ("KAF(GELU)", kaf), ("LibraKAN(GELU)", libra)]

    for name, model in models:
        train_model(model, x_tr, y_tr, epochs=epochs, device=device, amp=amp)
        with torch.no_grad():
            yhat = model(x_ev).cpu().squeeze(-1).numpy()
        preds.append((name, x_ev.cpu().squeeze(-1).numpy(), yhat))
    return x_ev.cpu().squeeze(-1).numpy(), y_ev.cpu().squeeze(-1).numpy(), preds

def kan_smooth_penalty(model, gamma=1e-3):
    if hasattr(model, "head"):
        W = model.head.weight
        diff2 = W[:, :-2] - 2*W[:, 1:-1] + W[:, 2:]
        return gamma * (diff2.pow(2).mean())
    return 0.0
# ============================================================
# 🖼️ 主程序绘图
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--save", type=str, default="fig_func_approx_extrap_colored.png")
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    use_cuda = (args.device.startswith("cuda") and torch.cuda.is_available())
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    set_seed(args.seed)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    fig.suptitle("Function Approximation Results", fontsize=18)

    # Panel 1: ReLU on sin
    x, y, preds = run_trio("sin", "relu", device, args.epochs, args.amp)
    ax = axes[0,0]
    plot_with_style(ax, x, y, "target")
    for name, xh, yh in preds:
        plot_with_style(ax, xh, yh, name)
    ax.set_title("ReLU Mixers on sin"); ax.legend(frameon=False)

    # Panel 2: ReLU on cos
    x, y, preds = run_trio("cos", "relu", device, args.epochs, args.amp)
    ax = axes[0,1]
    plot_with_style(ax, x, y, "target")
    for name, xh, yh in preds:
        plot_with_style(ax, xh, yh, name)
    ax.set_title("ReLU Mixers on cos"); ax.legend(frameon=False)

    # Panel 3: GELU on sin
    x, y, preds = run_trio("sin", "gelu", device, args.epochs, args.amp)
    ax = axes[1,0]
    plot_with_style(ax, x, y, "target")
    for name, xh, yh in preds:
        plot_with_style(ax, xh, yh, name)
    ax.set_title("GELU Mixers on sin"); ax.legend(frameon=False)

    # Panel 4: GELU on cos
    x, y, preds = run_trio("cos", "gelu", device, args.epochs, args.amp)
    ax = axes[1,1]
    plot_with_style(ax, x, y, "target")
    for name, xh, yh in preds:
        plot_with_style(ax, xh, yh, name)
    ax.set_title("GELU Mixers on cos"); ax.legend(frameon=False)

    for ax in axes.ravel():
        ax.set_xlim(-10, 10)
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(args.save, dpi=200)
    print(f"Saved figure to {args.save}")

# ============================================================
if __name__ == "__main__":
    main()
