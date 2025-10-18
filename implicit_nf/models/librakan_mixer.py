# models/librakan_mixer.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# ES window for NUFFT-like branch
# -------------------------
def es_window(freq: torch.Tensor, beta: float = 6.0) -> torch.Tensor:
    """
    Exponential-of-semicircle (ES) window for spectral kernel shaping.
    freq: (F, W) or broadcastable
    return: same shape as freq
    """
    denom = freq.detach().abs().amax()
    if float(denom) < 1e-8:
        return torch.ones_like(freq)
    u = freq / denom
    return torch.exp(-beta * (u ** 2))


# -------------------------
# Soft-thresholding (sparsity gate)
# -------------------------
class SoftThreshold(nn.Module):
    def __init__(self, tau: float = 1e-3):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(float(tau)), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # sign(x) * relu(|x| - tau)
        return torch.sign(x) * F.relu(x.abs() - self.tau)


# -------------------------
# LibraKAN Mixer (NUFFT-like spectral + local)
# -------------------------
class LibraKANMixer(nn.Module):
    def __init__(
            self,
            width: int,
            Fmax: int = None,
            rho: float = 0.3,
            spectral_scale: float = 0.8,
            es_beta: float = 6.0,
            l1_alpha: float = 1e-4,
            lambda_init: float = 0.01,
            lambda_trainable: bool = True,
            dropout: float = 0.0,
            use_layernorm: bool = True,
            use_cos_pair: bool = False,
            spectral_dropout: float = 0.0,
            p_sparse: float = 1.0,
            p_trainable: bool = False,
            p_min: float = 0.0,
            p_max: float = 1.0,
            lp_eps: float = 1e-6,
            p_reg_weight: float = 0.0,
            eps: float = 1e-8,
            tau_soft: float = 1e-3,
    ):
        super().__init__()
        self.width = int(width)
        self.F = int(Fmax or min(2 * width, 2048))
        self.rho = float(rho)
        self.spectral_scale = float(spectral_scale)
        self.es_beta = float(es_beta)
        self.l1_alpha = float(l1_alpha)
        self.spectral_dropout = float(spectral_dropout)
        self.use_cos_pair = bool(use_cos_pair)
        self.eps = float(eps)

        # learnable frequencies per channel: (F, W)
        base = torch.linspace(-math.pi, math.pi, self.F).view(self.F, 1).repeat(1, self.width)
        self.freq = nn.Parameter(base)  # (F, W)

        self.alpha = nn.Parameter(torch.randn(self.F, self.width) * 0.01)
        self.gate = nn.Parameter(torch.ones(self.F, self.width) * float(lambda_init), requires_grad=lambda_trainable)

        # soft-threshold
        self.soft_thr = SoftThreshold(tau=tau_soft)

        self._spec_dropout = nn.Dropout(self.spectral_dropout) if self.spectral_dropout > 1e-12 else nn.Identity()

        self.norm = nn.LayerNorm(self.width) if use_layernorm else nn.Identity()
        self.local = nn.Sequential(
            nn.Linear(self.width, self.width),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.p_min, self.p_max = float(p_min), float(p_max)
        self.lp_eps = float(lp_eps)
        self.p_reg_weight = float(p_reg_weight)

        def _inv_sigmoid(y: float) -> float:
            y = min(max(y, 1e-3), 1.0 - 1e-3)
            return math.log(y) - math.log(1 - y)

        p0_unit = (float(p_sparse) - self.p_min) / max(1e-8, (self.p_max - self.p_min))
        self.p_logit = nn.Parameter(torch.tensor(_inv_sigmoid(p0_unit), dtype=torch.float32),
                                    requires_grad=bool(p_trainable))

        self._h_local = None
        self._h_spec = None
        self._thr_w = None  

    def current_p(self) -> torch.Tensor:
        p_unit = torch.sigmoid(self.p_logit)
        return self.p_min + (self.p_max - self.p_min) * p_unit

    def spectral_params(self):
        return [self.freq, self.alpha, self.gate]

    def _lp_penalty(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.clamp(self.current_p(), 1e-3, 1.0)
        return torch.mean((x.abs() + self.lp_eps) ** p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, W)
        return: (B, W)
        """
        B, W = x.shape
        assert W == self.width, f"Input width {W} != mixer width {self.width}"

        x_n = self.norm(x)  # (B, W)
        h_local = self.local(x_n)  # (B, W)

        # --- spectral branch ---
        freq = self.freq * self.spectral_scale  # (F, W)
        win = es_window(freq, self.es_beta)  # (F, W)

        raw_w = self.alpha * self.gate  # (F, W)
        thr_w = self.soft_thr(raw_w) * win  # (F, W)
        thr_w = self._spec_dropout(thr_w)  # dropout on spectral coefficients

        basis_sin = torch.sin(x_n.unsqueeze(1) * freq)  # (B, F, W)
        h_spec = (basis_sin * thr_w.unsqueeze(0)).sum(dim=1)  # (B, W)

        if self.use_cos_pair:
            basis_cos = torch.cos(x_n.unsqueeze(1) * freq)  # (B, F, W)
            h_spec = h_spec + (basis_cos * thr_w.unsqueeze(0)).sum(dim=1)

        rms = torch.sqrt(h_spec.pow(2).mean(dim=1, keepdim=True) + self.eps)  # (B,1)
        h_spec = h_spec / rms

        out = h_local + self.rho * h_spec

        self._h_local = h_local
        self._h_spec = h_spec
        self._thr_w = thr_w

        return out

    def aux_losses(self):
        l_p_spec = self._lp_penalty(self._h_spec)
        p_reg = self.p_reg_weight * self.current_p()

        el = torch.norm(self._h_local, p=2, dim=-1).mean()
        es = torch.norm(self._h_spec, p=2, dim=-1).mean()
        balance = (el - es).pow(2)

        return {"l1_spec": l_p_spec + p_reg, "balance": balance}

    @torch.no_grad()
    def effective_alpha(self) -> torch.Tensor:
        freq = self.freq * self.spectral_scale
        win = es_window(freq, self.es_beta)
        raw_w = self.alpha * self.gate
        thr_w = self.soft_thr(raw_w) * win
        return thr_w

