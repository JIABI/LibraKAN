import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nufft_es import nufft_es_forward
from .shrinkage import generalized_shrinkage, p_from_plogit

class LibraKANLayer(nn.Module):
    """
    LibraKAN hybrid activation.
      φ_Libra(x) = a ⊙ GELU(W_l x + b_l) + b ⊙ V z,
      u = [Ω^T x] (and u2 = [Ω2^T x] when nufft_dim=2),
      z_raw = N(u[,u2]; ES), z = S_{λ,p}(z_raw), out = α·local + β·(V z).
    Options:
      - λ learnable or fixed
      - p learnable (via plogit) or fixed in (0,1)
      - NUFFT–ES in 1D or 2D (separable ES window)
    """
    def __init__(self,
                 in_dim, out_dim,
                 F=256, spectral_scale=1.0,
                 es_beta=6.0, es_fmax=None,
                 nufft_dim=1,
                 lambda_init=0.01, lambda_trainable=True,
                 p_fixed=0.5, p_trainable=True,
                 base_activation="gelu",
                 use_layernorm=False,
                 dropout=0.0):
        super().__init__()
        assert nufft_dim in (1, 2), "nufft_dim must be 1 or 2"
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.F = F
        self.spectral_scale = spectral_scale
        self.es_beta = es_beta
        self.es_fmax = es_fmax
        self.nufft_dim = nufft_dim

        # Local branch
        self.W_l = nn.Linear(in_dim, out_dim)
        self.use_layernorm = use_layernorm
        self.ln = nn.LayerNorm(out_dim) if use_layernorm else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.base_activation = base_activation

        # Spectral branch: frequency dictionaries
        self.Omega = nn.Parameter(torch.randn(in_dim, F) * (spectral_scale / math.sqrt(in_dim)))
        if nufft_dim == 2:
            self.Omega2 = nn.Parameter(torch.randn(in_dim, F) * (spectral_scale / math.sqrt(in_dim)))
        else:
            self.Omega2 = None

        # Per-frequency coefficients ε ∈ R^F
        self.epsilon = nn.Parameter(torch.randn(F) * 0.02)
        # Output mixing V ∈ R^{out_dim x F}
        self.V = nn.Parameter(torch.randn(out_dim, F) / math.sqrt(F))

        # Channel-wise fusion scales
        self.alpha = nn.Parameter(torch.ones(out_dim))
        self.beta = nn.Parameter(torch.ones(out_dim))

        # Shrinkage S_{λ,p}
        if lambda_trainable:
            self.lambda_param = nn.Parameter(torch.tensor(float(lambda_init)))
            self.lambda_fixed = None
        else:
            self.lambda_param = None
            self.lambda_fixed = float(lambda_init)

        self.p_trainable = bool(p_trainable)
        if self.p_trainable:
            self.plogit = nn.Parameter(torch.logit(torch.tensor(float(p_fixed)).clamp(1e-4, 1-1e-4)))
            self.p_fixed = None
        else:
            self.plogit = None
            self.p_fixed = float(p_fixed)

        # buffers for logging
        self.register_buffer("last_abs_z", None, persistent=False)  # |z| after shrinkage

    def _act(self, x):
        if self.base_activation == "gelu":
            return F.gelu(x)
        elif self.base_activation == "silu":
            return F.silu(x)
        else:
            return F.gelu(x)

    def forward(self, x):
        # x: (..., in_dim)
        orig_shape = x.shape
        x = x.view(-1, self.in_dim)

        # Local
        local = self._act(self.W_l(x))
        local = self.ln(local)
        local = self.dropout(local)

        # Spectral: phases
        u1 = x @ self.Omega  # (N,F)
        if self.nufft_dim == 2:
            u2 = x @ self.Omega2  # (N,F)
            u = torch.stack([u1, u2], dim=-1)  # (N,F,2)
        else:
            u = u1  # (N,F)

        # NUFFT–ES
        z_raw = nufft_es_forward(u, self.epsilon, beta=self.es_beta, fmax=self.es_fmax)
        # λ and p
        lam = self.lambda_param if self.lambda_param is not None else self.lambda_fixed
        if self.p_trainable:
            pval = p_from_plogit(self.plogit)
        else:
            pval = torch.tensor(self.p_fixed, device=z_raw.device, dtype=z_raw.dtype)
        # Shrinkage
        z = generalized_shrinkage(z_raw, lam, pval)
        # Log |z| for spectrum
        with torch.no_grad():
            self.last_abs_z = z.abs().detach()

        # Mixout
        spec = z @ self.V.t()
        out = self.alpha * local + self.beta * spec
        out = out.view(*orig_shape[:-1], self.out_dim)
        return out

    @torch.no_grad()
    def active_freq(self, thresh=1e-3):
        if self.last_abs_z is None:
            return 0
        active = (self.last_abs_z > thresh).any(dim=0).sum().item()
        return int(active)

class LibraKANBlock(nn.Module):
    def __init__(self, dim, hidden_dim, **kwargs):
        super().__init__()
        self.in_proj = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.libra = LibraKANLayer(hidden_dim, hidden_dim, **kwargs)
        self.out_proj = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        h = self.act(self.in_proj(x))
        h = self.libra(h)
        return self.out_proj(h)

def make_librakan_mixer(dim, hidden_dim, **libra_kwargs):
    return LibraKANBlock(dim, hidden_dim, **libra_kwargs)
