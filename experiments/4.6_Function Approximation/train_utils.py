
import math
import torch
import torch.nn.functional as F

@torch.no_grad()
def set_seed(seed: int = 41):
    import numpy as np, random
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def make_data(n_train=512, n_eval=2048, fn="sin", train_range=(0.0,1.0), eval_range=(-20.0,20.0), device="cpu", dtype=torch.float32):
    x_tr = torch.linspace(train_range[0], train_range[1], n_train, device=device, dtype=torch.float32).unsqueeze(-1)
    x_ev = torch.linspace(eval_range[0],  eval_range[1],  n_eval,  device=device, dtype=torch.float32).unsqueeze(-1)
    if fn == "sin":
        y_tr = torch.sin(2*math.pi*x_tr)
        y_ev = torch.sin(2*math.pi*x_ev)
    else:
        y_tr = torch.cos(2*math.pi*x_tr)
        y_ev = torch.cos(2*math.pi*x_ev)
    return x_tr.to(dtype), y_tr.to(dtype), x_ev.to(dtype), y_ev.to(dtype)

def build_param_groups(model, lr_main=5e-3, lr_spec=2e-3, wd_main=1e-3):
    spec_keys = ["freqs_raw","phase","lam_raw","p_raw","es_beta_raw","gate"]
    pg_spec, pg_main = [], []
    for n,p in model.named_parameters():
        if any(k in n for k in spec_keys):
            pg_spec.append(p)
        else:
            pg_main.append(p)
    groups = [
        {"params": pg_spec, "lr": lr_spec, "weight_decay": 0.0},
        {"params": pg_main, "lr": lr_main, "weight_decay": wd_main},
    ]
    return groups
def kan_smooth_penalty(model, gamma=5e-3):
    if hasattr(model, "head"):
        W = model.head.weight
        diff2 = W[:, :-2] - 2*W[:, 1:-1] + W[:, 2:]
        return gamma * (diff2.pow(2).mean())
    return 0.0

def train_model(model, x, y, epochs=1000, device="cpu", lr_main=5e-3, lr_spec=2e-3, wd_main=1e-3, amp=False, extras_weight=None):
    model.to(device)
    x = x.to(device)
    y = y.to(device)
    param_groups = build_param_groups(model, lr_main, lr_spec, wd_main)
    opt = torch.optim.AdamW(param_groups, betas=(0.9,0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    loss_hist = []
    for t in range(epochs):
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            yhat = model(x)
            loss = F.mse_loss(yhat, y)
            if "KAN" in model.__class__.__name__:
                loss = loss + kan_smooth_penalty(model, gamma=5e-3)
            if hasattr(model, "extra_losses"):
                extras = model.extra_losses()
                if extras_weight is None:
                    alpha_gl = 1e-4; beta_decor = 1e-4
                else:
                    alpha_gl, beta_decor = extras_weight
                if "group_lasso" in extras: loss = loss + alpha_gl * extras["group_lasso"]
                if "decor" in extras:       loss = loss + beta_decor * extras["decor"]
                if "l1_spectral" in extras: loss = loss + extras["l1_spectral"]
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(opt); scaler.update()
        sched.step()
        if (t+1) % max(1, epochs//10) == 0:
            loss_hist.append((t+1, float(loss.detach().cpu())))
    return loss_hist
