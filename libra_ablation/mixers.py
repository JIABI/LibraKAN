import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8

def es_window(freq: torch.Tensor, beta: float = 6.0) -> torch.Tensor:
    denom = freq.detach().abs().amax()
    if float(denom) < EPS:
        return torch.ones_like(freq)
    u = freq / denom
    out = torch.exp(-beta * (u ** 2))
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

class GeneralizedShrink(nn.Module):
    def __init__(self, tau: float = 1e-3, learnable_tau: bool = False, p_sparse: float = 1.0):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(float(tau)), requires_grad=learnable_tau)
        self.p_sparse = float(p_sparse)
    def forward(self, z: torch.Tensor):
        p = float(self.p_sparse)
        absz = z.abs() + EPS
        thresh = self.tau * (absz ** max(0.0, 1.0 - p))
        shrunk = torch.sign(z) * F.relu(absz - thresh)
        return torch.nan_to_num(shrunk, nan=0.0, posinf=0.0, neginf=0.0)

class MLPBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(width, width)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class KAFBlock(nn.Module):
    def __init__(self, width: int, F: int = None, scale: float = 1.0, dropout: float = 0.0, uniform: bool = True):
        super().__init__()
        self.width = width
        self.F = F or min(2*width, 2048)
        self.scale = float(scale)
        self.dropout = nn.Dropout(dropout)
        if uniform:
            freq = torch.linspace(-math.pi, math.pi, self.F).view(self.F, 1).repeat(1, width)
        else:
            grid = torch.linspace(-math.pi, math.pi, self.F).view(self.F, 1).repeat(1, width)
            jitter = 0.1 * torch.sin(3.0 * grid)
            freq = grid + jitter
        self.register_buffer("freq", freq * self.scale)
        self.alpha = nn.Parameter(torch.randn(self.F, width) * 0.01)
        self.norm = nn.LayerNorm(width)
        self._thr_w = None

    def forward(self, x):
        x = self.norm(x)
        proj = x.unsqueeze(1) * self.freq
        feat = torch.sin(proj) + torch.cos(proj)
        feat = self.dropout(feat)
        w = self.alpha
        self._thr_w = w
        h = (feat * w.unsqueeze(0)).sum(dim=1)
        return torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)

    def active_freq(self, tau: float = 1e-3) -> int:
        w = self._thr_w if self._thr_w is not None else self.alpha
        return int((w.detach().abs() > tau).sum().item())

class LibraKAN(nn.Module):
    def __init__(
        self,
        width: int,
        Fmax: int = None,
        rho: float = 0.25,
        spectral_scale: float = 1.0,
        es_beta: float = 6.0,
        tau_soft: float = 1e-3,
        tau_learnable: bool = False,
        dropout: float = 0.0,
        use_layernorm: bool = True,
        use_cos_pair: bool = True,
        spectral_dropout: float = 0.0,
        no_soft: bool = False,
        uniform_freq: bool = False,
        no_es: bool = False,
        p_sparse: float = 1.0,
        gate_init: float = 0.01,
        alpha_init: float = 0.01,
        max_freq_scale: float = 2.0,
    ):
        super().__init__()
        self.width = width
        self.F = int(Fmax or min(1*width, 2048))
        self.rho = float(rho)
        self.spectral_scale = float(spectral_scale)
        self.es_beta = float(es_beta)
        self.no_soft = bool(no_soft)
        self.uniform_freq = bool(uniform_freq)
        self.no_es = bool(no_es)
        self.use_cos_pair = bool(use_cos_pair)
        self.p_sparse = float(p_sparse)
        self.max_freq_scale = float(max_freq_scale)

        base = torch.linspace(-math.pi, math.pi, self.F).view(self.F, 1).repeat(1, width)
        if self.uniform_freq:
            self.register_buffer("freq", base)
        else:
            self.freq = nn.Parameter(base)

        self.alpha = nn.Parameter(torch.randn(self.F, width) * alpha_init)
        self.gate  = nn.Parameter(torch.ones(self.F, width) * gate_init, requires_grad=True)

        self.shrink = GeneralizedShrink(tau=tau_soft, learnable_tau=tau_learnable, p_sparse=self.p_sparse)
        self._spec_dropout = nn.Dropout(spectral_dropout) if spectral_dropout > 1e-12 else nn.Identity()

        self.norm = nn.LayerNorm(width) if use_layernorm else nn.Identity()
        self.local = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout))

        self._h_local = None
        self._h_spec  = None
        self._thr_w   = None

    def forward(self, x):

        # -------- Local 分支 --------

        x_n = self.norm(x)

        # 关键：先把用于频谱分支的输入有界化，避免极端值

        x_spec = torch.tanh(x_n)  # 有界于 [-1,1]

        h_local = self.local(x_n)

        h_local = torch.nan_to_num(h_local, nan=0.0, posinf=0.0, neginf=0.0)

        # -------- Spectral 分支 --------

        # 安全频率缩放

        scl = max(min(float(self.spectral_scale), self.max_freq_scale), 0.1)

        freq = (self.freq * scl)

        win = torch.ones_like(freq) if self.no_es else es_window(freq, self.es_beta)

        # 阈后权重

        raw_w = self.alpha * self.gate

        if self.no_soft:

            thr_w = raw_w

        else:

            self.shrink.p_sparse = self.p_sparse

            thr_w = self.shrink(raw_w)

        # 加窗 + 硬裁剪（彻底防爆）

        thr_w = (thr_w * win).clamp_(-0.5, 0.5)

        thr_w = torch.nan_to_num(thr_w, nan=0.0, posinf=0.0, neginf=0.0)

        # 相位投影（用有界 x_spec）

        proj = x_spec.unsqueeze(1) * freq  # (N,F,W)

        s_term = torch.sin(proj)

        if self.use_cos_pair:
            s_term = s_term + torch.cos(proj)

        h_spec = (s_term * thr_w.unsqueeze(0)).sum(dim=1)

        # 能量归一 + 保险

        denom = torch.clamp(h_spec.pow(2).mean(dim=1, keepdim=True).sqrt(), min=1e-4)

        h_spec = h_spec / denom

        h_spec = torch.nan_to_num(h_spec, nan=0.0, posinf=0.0, neginf=0.0)

        # 融合 + 保险

        out = h_local + self.rho * h_spec

        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

        # 缓存给 Active-Freq/正则

        self._h_local = h_local

        self._h_spec = h_spec

        self._thr_w = thr_w

        return out

    def extra_losses(self, alpha_l1=0.0, beta_bal=0.0, use_hspec=True):
        spec = self._h_spec if use_hspec and (self._h_spec is not None) else self._thr_w
        l_spars = torch.tensor(0.0, device=self.alpha.device)
        if spec is not None and alpha_l1 > 0:
            p = max(1e-6, self.p_sparse)
            l_spars = (spec.abs().pow(p)).mean() * alpha_l1
        l_bal = torch.tensor(0.0, device=self.alpha.device)
        if beta_bal > 0 and (self._h_local is not None) and (self._h_spec is not None):
            l_bal = ((self._h_local.norm(dim=1) - self._h_spec.norm(dim=1))**2).mean() * beta_bal
        return {"l_sparse": l_spars, "l_balance": l_bal}

    def active_freq(self, tau: float = 1e-3) -> int:
        w = self._thr_w if self._thr_w is not None else (self.alpha * self.gate)
        return int((w.detach().abs() > tau).sum().item())

class MixerFFN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, kind="mlp", **kwargs):
        super().__init__()
        if kind == "mlp":
            core = MLPBlock(hidden)
        elif kind == "kaf":
            core = KAFBlock(hidden, F=kwargs.get("Fmax"), scale=kwargs.get("spectral_scale", 1.0),
                            dropout=kwargs.get("dropout", 0.0), uniform=kwargs.get("uniform_freq", True))
        elif kind == "librakan":
            core = LibraKAN(hidden, **kwargs)
        else:
            raise ValueError(f"Unknown mixer kind: {kind}")
        self.core = core
        self.proj_in = nn.Linear(in_dim, hidden)
        self.act = nn.GELU()
        self.proj_out = nn.Linear(hidden, out_dim)
    def forward(self, x):
        x = self.proj_in(x)
        x = self.core(x)
        x = self.act(x)
        x = self.proj_out(x)
        return x
