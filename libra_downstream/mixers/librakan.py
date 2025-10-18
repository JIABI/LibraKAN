import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def es_window(freq: torch.Tensor, beta: float = 6.0) -> torch.Tensor:
    denom = freq.detach().abs().amax()
    if float(denom) < 1e-8:
        return torch.ones_like(freq)
    u = freq / denom
    return torch.exp(-beta * (u ** 2))

class SoftThreshold(nn.Module):
    def __init__(self, tau: float = 1e-3):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(float(tau)), requires_grad=False)
    def forward(self, x):
        return torch.sign(x) * F.relu(x.abs() - self.tau)

class LibraKANBlock(nn.Module):
    def __init__(
        self,
        width: int,
        Fmax: int = None,
        rho: float = 0.25,
        spectral_scale: float = 0.8,
        es_beta: float = 6.0,
        tau_soft: float = 1e-3,
        dropout: float = 0.0,
        use_layernorm: bool = True,
        use_cos_pair: bool = False,
        spectral_dropout: float = 0.0,
        f_chunk: int = 256,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.width = width
        self.F = int(Fmax or min(2*width, 2048))
        self.rho = float(rho)
        self.spectral_scale = float(spectral_scale)
        self.es_beta = float(es_beta)
        self.f_chunk = int(f_chunk)
        self.eps = float(eps)

        base = torch.linspace(-math.pi, math.pi, self.F).view(self.F, 1).repeat(1, width)
        self.freq = nn.Parameter(base)  # (F,W)
        self.alpha = nn.Parameter(torch.randn(self.F, width) * 0.01)
        self.gate  = nn.Parameter(torch.ones(self.F, width) * 0.01, requires_grad=True)

        self.soft_thr = SoftThreshold(tau=tau_soft)
        self._spec_dropout = nn.Dropout(spectral_dropout) if spectral_dropout > 1e-12 else nn.Identity()
        self.use_cos_pair = bool(use_cos_pair)

        self.norm = nn.LayerNorm(width) if use_layernorm else nn.Identity()
        self.local = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout))

        self._h_local = None
        self._h_spec  = None
        self._thr_w   = None

    def forward(self, x):
        x_n = self.norm(x)
        h_local = self.local(x_n)

        freq = self.freq * self.spectral_scale
        win  = es_window(freq, self.es_beta)
        raw_w = self.alpha * self.gate
        thr_w = self.soft_thr(raw_w) * win
        thr_w = self._spec_dropout(thr_w)

        N, W = x_n.shape
        h_spec = torch.zeros_like(x_n)
        Ftot = freq.size(0)
        step = self.f_chunk if self.f_chunk > 0 else Ftot

        for s in range(0, Ftot, step):
            e = min(s + step, Ftot)
            freq_c = freq[s:e, :]    # (f, W)
            tw_c   = thr_w[s:e, :]   # (f, W)
            basis_s = torch.sin(x_n.unsqueeze(1) * freq_c)  # (N,f,W)
            h_spec = h_spec + (basis_s * tw_c.unsqueeze(0)).sum(dim=1)
            del basis_s
            if self.use_cos_pair:
                basis_c = torch.cos(x_n.unsqueeze(1) * freq_c)
                h_spec = h_spec + (basis_c * tw_c.unsqueeze(0)).sum(dim=1)
                del basis_c

        rms = torch.sqrt(h_spec.pow(2).mean(dim=1, keepdim=True) + self.eps)
        h_spec = h_spec / rms
        out = h_local + self.rho * h_spec

        self._h_local = h_local
        self._h_spec  = h_spec
        self._thr_w   = thr_w
        return out

class MixerFFN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, **kwargs):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden)
        self.mixer = LibraKANBlock(hidden, **kwargs)
        self.act = nn.GELU()
        self.proj_out = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = self.proj_in(x)
        x = self.mixer(x)
        x = self.act(x)
        x = self.proj_out(x)
        return x
