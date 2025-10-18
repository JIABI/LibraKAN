import os
import argparse
import time
import math
import csv
import random
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

# --- project utils ---
from utils import make_coord_grid, image_to_tensor01, save_image01, set_seed_all, cosine_lr, crop_visuals
from metrics import psnr_torch, ssim_torch

# --- model registry ---
from models.mlp import ImplicitMLP
from models.librakan_wrap import LibraKANImplicit
try:
    from models.kaf_wrap import KAFImplicit
    HAS_KAF = True
except Exception as e:
    HAS_KAF = False

REGISTRY = {
    "MLP":      lambda W, L, C, kwargs: ImplicitMLP(in_dim=2, hidden=W, depth=L, out_dim=C, act="gelu"),
    "LibraKAN": lambda W, L, C, kwargs: LibraKANImplicit(in_dim=2, hidden=W, depth=L, out_dim=C, **kwargs),
}
if HAS_KAF:
    REGISTRY["KAF"] = lambda W, L, C, kwargs: KAFImplicit(in_dim=2, hidden=W, depth=L, out_dim=C, F=max(16, 2*W))

def build_model(name: str, W: int, L: int, C: int, libra_kwargs: Dict[str, Any]) -> nn.Module:
    if name not in REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(REGISTRY.keys())}")
    return REGISTRY[name](W, L, C, libra_kwargs)

def split_train_val_mask(H: int, W: int, val_ratio: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    mask = rng.random((H, W)) < val_ratio
    return mask

def sample_batch(coords: torch.Tensor, target: torch.Tensor, batch_size: int):
    # coords: (H*W,2), target: (H*W,C)
    N = coords.shape[0]
    idx = torch.randint(0, N, (batch_size,), device=coords.device)
    return coords[idx], target[idx]

def steps_to_threshold(psnr_hist, threshold: float) -> int:
    # psnr_hist: list of (step, psnr)
    for step, val in psnr_hist:
        if val >= threshold:
            return step
    return -1

def train_one_image(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    set_seed_all(args.seed)

    # --- load image ---
    img = Image.open(args.image).convert("RGB" if not args.gray else "L")
    img_np = np.asarray(img)
    H, W = img_np.shape[:2]
    C = 3 if (not args.gray) else 1

    # normalize to [0,1]
    img_t = image_to_tensor01(img_np).to(device)  # (1,C,H,W)
    coords = make_coord_grid(H, W, device=device) # (H*W,2) in [-1,1]^2
    target = img_t.view(1, C, -1).permute(0, 2, 1).reshape(-1, C)  # (H*W, C)

    # split a small val set of pixels for early stopping
    val_mask_hw = split_train_val_mask(H, W, val_ratio=0.05, seed=args.seed)
    val_mask = torch.from_numpy(val_mask_hw.reshape(-1)).to(device)
    tr_mask = ~val_mask
    coords_tr, coords_val = coords[tr_mask], coords[val_mask]
    target_tr, target_val = target[tr_mask], target[val_mask]

   
    libra_kwargs = dict(
        Fmax=min(int(2 * args.width), 1578),
        p_sparse = 1.0,
        p_trainable = True,
        p_min = 0.0, p_max = 1.0,
        lp_eps = 1e-6,
        p_reg_weight = 0.0,
        spectral_scale=0.9,  
        es_beta=6.0,           
        lambda_init=0.01,
        lambda_trainable=True,  
        l1_alpha=5e-5, 
        dropout=0.0,
        use_layernorm=True,      
        use_cos_pair=False,       
        spectral_dropout=0.02,    
        rho=0.25,                 
        tau_soft = 1e-3
    )
    model = build_model(args.model, args.width, args.depth, C, libra_kwargs).to(device)

    def set_p_trainable(model, flag: bool):
        for m in model.modules():
            if hasattr(m, "p_logit"):
                p = m.p_logit
                if isinstance(p, torch.Tensor):
                    p.requires_grad_(bool(flag))

    def set_soft_tau(model, value: float):
        for m in model.modules():
            if hasattr(m, "soft_thr"):  
                m.soft_thr.tau.data.fill_(float(value))

    def set_spectral_scale(model, value: float):
        for m in model.modules():
            if hasattr(m, "spectral_scale"): 
                m.spectral_scale = float(value)

    def set_rho(model, value: float):
        for m in model.modules():
            if hasattr(m, "rho"):
                m.rho = float(value)

    def freeze_gate(model, freeze: bool = True):
        for m in model.modules():
            if hasattr(m, "gate"):
                m.gate.requires_grad_(not freeze)
                if freeze and m.gate.grad is not None:
                    m.gate.grad = None
    def set_p_sparse(model, value: float):
        for m in model.modules():
            if hasattr(m, "p_sparse"):
                m.p_sparse = float(value)

    def freeze_spectral(model, freeze: bool = True):
        for m in model.modules():
            if hasattr(m, "spectral_params"):
                for p in m.spectral_params():
                    p.requires_grad_(not freeze)
                    if freeze and getattr(p, "grad", None) is not None:
                        p.grad = None

 
    def set_group_lr(optim, name: str, lr: float, sched=None):
        for i, pg in enumerate(optim.param_groups):
            if pg.get("name") == name:
                pg["lr"] = float(lr)
                if sched is not None and hasattr(sched, "base_lrs") and i < len(sched.base_lrs):
                    sched.base_lrs[i] = float(lr)
    def _inv_sigmoid(y:float):
        y = max(min(y, 1-1e-3), 1e-3)
        return math.log(y) - math.log(1-y)
    def set_p_value(model, value: float):
        for m in model.modules():
            if hasattr(m, "p_logit"):
                pmin =getattr(m, "p_min", 0.0)
                pmax = getattr(m, "p_max", 1.0)
                u = (float(value)-pmin)/max(1e-8, (pmax-pmin))
                u = max(min(u, 1-1e-3), 1e-3)
                m.p_logit.data.copy_(torch.tensor(_inv_sigmoid(u), device=m.p_logit.device))


    # --- optimizer with param groups (LibraKAN: smaller lr for spectral params) ---
    if hasattr(model, "modules"):
        spec_params, loc_params, other_params = [], [], []
        for m in model.modules():
            if hasattr(m, "spectral_params"):
                spec_params += list(m.spectral_params())
            if hasattr(m, "local_params"):
                loc_params += list(m.local_params())
        all_ids = set(id(p) for p in spec_params + loc_params)
        for p in model.parameters():
            if id(p) not in all_ids:
                other_params.append(p)
        p_params= [m.p_logit for m in model.modules() if hasattr(m, "p_logit")]

        param_groups = []
        if p_params:
            param_groups.append({"params": p_params, "lr": 1e-4, "name": "p"})  # 小 lr
        if loc_params:
            param_groups.append({"params": loc_params, "lr": args.lr, "name": "local"})
        if spec_params:
            param_groups.append({"params": spec_params, "lr": max(args.lr * 0.4, 1e-4), "name": "spectral"})  # 先 0.2x
        optim = torch.optim.Adam(param_groups, betas=(0.9, 0.999))

    else:
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- scheduler ---
    #sched = cosine_lr(optim, args.lr, args.lr_min, args.steps)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.steps, eta_min=args.lr_min)

    # --- training ---
    run_dir = os.path.join(
        args.out_dir,
        os.path.splitext(os.path.basename(args.image))[0] + f"_{args.model.lower()}"
    )
    os.makedirs(run_dir, exist_ok=True)

    best_val_psnr = -1e9
    best_state = None
    last_improve = 0
    hist = []

    model.train()
    t0 = time.time()
    bar = tqdm(range(1, args.steps + 1), desc=f"Train {args.model} ({H}x{W})", dynamic_ncols=True)
    running_loss = 0.0
    last_eval_psnr, last_eval_ssim = None, None
    last_l1, last_bal = 0.0, 0.0

    for step in bar:
        optim.zero_grad(set_to_none=True)

        # ===== 2-phase warm start (放在每步开始处) =====
        if step == 1:
            # 起步：谱分支“只前向”，更稀疏、更低频、更小权重
            set_rho(model, 0.15)
            set_soft_tau(model, 2e-3)  # 更稀疏（抗抖）
            set_spectral_scale(model, 0.8)  # 低一点的频率尺度
            freeze_spectral(model, True)  # 冻结谱参数
            freeze_gate(model, True)  # 冻结 gate(lambda)
            # 谱参数组 LR 先小或为 0（看你想完全不训还是微训）
            set_group_lr(optim, "spectral", 0.0, sched)
            #set_p_sparse(model, 1.0)
            set_p_value(model, 1)
            set_p_trainable(model, False)

        if step == 5000:
            # 放开：逐步启用谱分支学习，增加权重，放松稀疏阈值
            set_rho(model, 0.20)
            set_soft_tau(model, 1e-3)  # 放松稀疏，允许更多有用频率激活
            set_spectral_scale(model, 0.85)  # 略升频率
            freeze_spectral(model, False)  # 开训谱参数
            freeze_gate(model, False)  # 开训 gate
            #set_group_lr(optim, "spectral", max(args.lr * 0.4, 1e-4), sched)
            set_p_sparse(model, 0.5)
            set_p_trainable(model, False)
            set_group_lr(optim, "spectral", 1e-4, sched)


        if step == 10000:
            # 完全体：强化边缘细节（建筑场景）
            set_rho(model, 0.25)
            set_spectral_scale(model, 0.90)  # 再升一点频率
            # 若前期用 L1=1e-4，可在此后把 reg 降一点（避免过抑制细节）
            for m in model.modules():
                if hasattr(m, "l1_alpha"):
                    m.l1_alpha = 5e-5

        # 可选：在 60% 进度（例如 0.6 * steps）再轻微放开
        if step == int(0.6 * args.steps):
            set_spectral_scale(model, 0.95)
        if step % args.eval_every == 0 :
            with torch.no_grad():
                ps = []
                for m in model.modules():
                    if hasattr(m, "current_p"):
                        ps.append(float(m.current_p().mean().item()))
                if ps:
                    print(f"[Debug] mean(p)={sum(ps)/len(ps):.3f}")
        # ===== 2-phase end =====

        # sample pixels
        c_b, y_b = sample_batch(coords_tr, target_tr, args.batch)
        pred = model(c_b)  # (B,C)
        #loss = F.mse_loss(pred, y_b)
        extra = model.extra_losses()  # 聚合各层的 aux_losses()
        loss = F.mse_loss(pred, y_b) + args.alpha_sparse * extra["l1_spectral"] \
               + args.alpha_balance * extra["balance"]

        # extra losses (LibraKAN only; MLP/KAF 则返回 0)
        l1_term = torch.zeros((), device=device)
        bal_term = torch.zeros((), device=device)
        if hasattr(model, "extra_losses"):
            extra = model.extra_losses()
            l1_term = extra.get("l1_spectral", torch.zeros((), device=device))
            bal_term = extra.get("balance", torch.zeros((), device=device))
            loss = loss + args.alpha_sparse * l1_term + args.alpha_balance * bal_term

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        sched.step()

        running_loss += float(loss.item())
        last_l1 = float(l1_term.item()) if torch.is_tensor(l1_term) else 0.0
        last_bal = float(bal_term.item()) if torch.is_tensor(bal_term) else 0.0
        lr_now = optim.param_groups[0]["lr"]

        postfix = {
            "loss": f"{running_loss / step:.4f}",
            "lr": f"{lr_now:.2e}",
            "l1": f"{last_l1:.2e}",
            "bal": f"{last_bal:.2e}",
        }
        if last_eval_psnr is not None:
            postfix.update({"PSNR(val)": f"{last_eval_psnr:.2f}dB"})
        bar.set_postfix(postfix)

        # periodic val
        if step % args.eval_every == 0 or step == args.steps:
            with torch.no_grad():
                # full image predict
                pred_full = []
                BS_eval = 1 << 16
                for i in range(0, coords.shape[0], BS_eval):
                    pred_full.append(model(coords[i:i + BS_eval]))
                pred_full = torch.cat(pred_full, dim=0).view(H, W, C).permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)
                psnr_val = psnr_torch(pred_full, img_t)
                ssim_val = ssim_torch(pred_full, img_t)

                last_eval_psnr = float(psnr_val.item())
                last_eval_ssim = float(ssim_val.item())
                hist.append((step, last_eval_psnr))

                improved = False
                if psnr_val > best_val_psnr + 1e-5:
                    best_val_psnr = float(psnr_val.item())
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    last_improve = step
                    improved = True

                bar.write(
                    f"[Eval] step {step:>6d} | PSNR {last_eval_psnr:.2f} dB | "
                    f"SSIM {last_eval_ssim:.4f} | best {best_val_psnr:.2f} dB | "
                    f"l1 {last_l1:.2e} bal {last_bal:.2e} | lr {lr_now:.2e}"
                )

                # 早停（如需启用，取消注释）
                # if step - last_improve >= args.early_stop_patience:
                #     bar.write(f"[EarlyStop] no improvement for {args.early_stop_patience} steps, stop at {step}.")
                #     break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # final eval
    with torch.no_grad():
        pred_full = []
        BS_eval = 1 << 18
        for i in range(0, coords.shape[0], BS_eval):
            pred_full.append(model(coords[i:i + BS_eval]))
        pred_full = torch.cat(pred_full, dim=0).view(H, W, C).permute(2, 0, 1).unsqueeze(0)
        psnr = psnr_torch(pred_full, img_t).item()
        ssim = ssim_torch(pred_full, img_t).item()
        step_at = steps_to_threshold(hist, args.psnr_threshold)

    # save images
    save_image01(img_t, os.path.join(run_dir, "target.png"))
    save_image01(pred_full, os.path.join(run_dir, f"pred_{args.model.lower()}.png"))
    crop_visuals(img_t, pred_full, os.path.join(run_dir, "crops.png"))

    # Active-Freq statistics
    active_freq = -1
    if hasattr(model, "active_freq"):
        try:
            active_freq = int(model.active_freq(tau=args.tau_active))
        except Exception:
            active_freq = -1  # unavailable

    # log csv
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "results_single_image_nf.csv")
    header = ["dataset", "image", "model", "W", "L", "params", "psnr", "ssim", "steps_at_threshold", "active_freq", "seed"]
    params = sum(p.numel() for p in model.parameters())
    row = [args.dataset, os.path.basename(args.image), args.model, args.width, args.depth, params, psnr, ssim, step_at, active_freq, args.seed]
    write_header = (not os.path.exists(csv_path))
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header: w.writerow(header)
        w.writerow(row)

    # save report json
    report = {
        "dataset": args.dataset,
        "image": os.path.basename(args.image),
        "H": H, "W": W, "C": C,
        "model": args.model,
        "W_hidden": args.width, "L_layers": args.depth,
        "psnr": psnr, "ssim": ssim,
        "steps_at_threshold": step_at,
        "best_val_psnr": best_val_psnr,
        "active_freq": active_freq,
        "seed": args.seed,
        "time_sec": time.time() - t0,
    }
    with open(os.path.join(run_dir, "report.json"), "w") as f:
        import json; json.dump(report, f, indent=2)

    print(f"[Done] {args.model} {os.path.basename(args.image)}  PSNR={psnr:.2f}dB  SSIM={ssim:.4f}  ActiveFreq={active_freq}  Steps@{args.psnr_threshold}dB={step_at}")
    print(f"Saved to: {run_dir}")
    return report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, default="/home/ubuntu/PycharmProjects/LiberNet/data/Urban 100/X4 Urban100/X4/HIGH x4 URban100/img_053_SRF_4_HR.png")
    ap.add_argument("--dataset", type=str, default="custom")
    ap.add_argument("--out_dir", type=str, default="implicit_nf/runs_nf")

    ap.add_argument("--model", type=str, choices=["MLP","LibraKAN","KAF"], default="MLP")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--gray", action="store_true",  help="force grayscale")

    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_min", type=float, default=1e-4)
    ap.add_argument("--eval_every", type=int, default=5000)
    ap.add_argument("--psnr_threshold", type=float, default=30.0)
    ap.add_argument("--early_stop_patience", type=int, default=50000)

    ap.add_argument("--alpha_sparse", type=float, default=1.0)
    ap.add_argument("--alpha_balance", type=float, default=0.1)
    ap.add_argument("--l1_alpha", type=float, default=1e-5)
    ap.add_argument("--tau_active", type=float, default=1e-3)

    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--cpu", action="store_true")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train_one_image(args)

if __name__ == "__main__":
    main()
