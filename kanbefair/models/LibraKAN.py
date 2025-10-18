# models/librakan_ext.py
from typing import Sequence, Tuple, Union, Optional, Dict, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Utils
# -----------------------------
def _act_module(name: str) -> nn.Module:
    name = name.lower()
    if name == "gelu": return nn.GELU()
    if name in ["silu", "swish"]: return nn.SiLU(inplace=True)
    if name == "relu": return nn.ReLU(inplace=True)
    if name in ["identity", "linear"]: return nn.Identity()
    raise ValueError(f"Unsupported activation: {name}")

def soft_threshold(x: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    # lam can be scalar tensor or broadcastable vector
    return torch.sign(x) * torch.clamp(torch.abs(x) - lam, min=0.0)

def _has_nufft() -> Optional[str]:
    try:
        import torchkbnufft  # noqa: F401
        return "torchkbnufft"
    except Exception:
        try:
            import torchnufft  # noqa: F401
            return "torchnufft"
        except Exception:
            return None

def _es_window(freq_abs: torch.Tensor, fmax: float, beta: float = 8.0) -> torch.Tensor:
    """
    Exponential-of-semicircle (ES) window (simplified 1D radial form).
    w(|f|) = exp(beta * (sqrt(1 - (|f|/fmax)^2) - 1)), for |f| <= fmax; 0 otherwise.
    """
    r = (freq_abs / (fmax + 1e-6)).clamp(min=0.0, max=1.0)
    inside = (1.0 - r * r).clamp(min=0.0)
    w = torch.exp(beta * (torch.sqrt(inside) - 1.0))
    w = torch.where(r <= 1.0, w, torch.zeros_like(w))
    return w

def _device_of(t: torch.Tensor) -> torch.device:
    try:
        return t.device
    except Exception:
        return torch.device("cpu")

# -----------------------------
# LibraKAN Layer (dual-branch)
# -----------------------------
class LibraKANLayer(nn.Module):
    """
    LibraKAN dual-path layer:
      y = GELU(W_l x + b_l) + S_lambda( SpectralBranch(x) )

    SpectralBranch:
      - Prefer NUFFT if available; otherwise learnable non-uniform Fourier features:
            z_k = x @ Ω_k + φ_k
            ψ = [cos(z), sin(z)] (2F) --(gate & ES)--> proj_spec -> R^{D_out}
      - ES window for spectral localization (weight frequencies)
      - Soft-threshold for sparsity; optional learnable lambda
      - L1 penalty on spectral response after threshold (exposed via .sparsity_loss)
      - Channel gate on 2F features for “effective S” & pruning

    Args:
      dim_in, dim_out: feature dims
      F:      max number of non-uniform frequencies
      spectral_scale: init scale of Ω
      es_beta, es_fmax: ES window params (fmax>0)
      lambda_init: initial soft-threshold
      lambda_trainable: whether λ is learnable
      l1_alpha: weight for L1(spectral) (can be 0 to disable)
      base_activation: local branch activation ("gelu"/"silu"/...)
      use_layernorm: apply LayerNorm on inputs
      dropout: dropout after fusion
    """
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        *,
        F: int = 128,
        spectral_scale: float = 1.0,
        es_beta: float = 8.0,
        es_fmax: Optional[float] = None,
        lambda_init: float = 0.01,
        lambda_trainable: bool = True,
        l1_alpha: float = 1e-4,
        base_activation: str = "gelu",
        use_layernorm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.F = F
        self.es_beta = es_beta
        # 若未给 es_fmax，用 spectral_scale * sqrt(dim_in) 做一个稳健上界
        self.es_fmax = es_fmax if es_fmax is not None else spectral_scale * math.sqrt(max(1, dim_in))
        self.l1_alpha = l1_alpha
        self.nufft_backend = _has_nufft()

        # ---- Local (Linear -> GELU) ----
        self.local = nn.Linear(dim_in, dim_out, bias=True)
        self.base_act = _act_module(base_activation)

        # ---- Spectral (learnable non-uniform features) ----
        # 对数均匀初始化 + 轻微扰动（聚焦低频+少量谐波）
        low, high = 0.5, 6.0
        lin = torch.linspace(math.log(low + 1e-6), math.log(high), self.F).exp()  # (F,)
        omega_1d = lin + 0.05 * torch.randn(self.F)  # 轻微扰动
        Omega = torch.zeros(dim_in, self.F)
        Omega[0, :] = omega_1d  # 1D 主导；多维可按需分配
        self.Omega = nn.Parameter(Omega * (spectral_scale / 1.0))
        self.Phi = nn.Parameter(torch.empty(self.F).uniform_(0.0, 2.0 * math.pi))  # 相位
        self.freq_gate = nn.Parameter(torch.ones(self.F))                           # 每频率门

        # 2F 通道级门控（cos+sin），用于“有效频数 S”裁剪与统计
        self.spec_gate = nn.Parameter(torch.ones(2 * self.F))

        # 投影回 dim_out
        self.proj_spec = nn.Linear(2 * self.F, dim_out, bias=False)
        nn.init.xavier_uniform_(self.proj_spec.weight)

        # Soft-threshold λ（标量），可学习
        lam = torch.tensor(lambda_init, dtype=torch.float32)
        self.lam = nn.Parameter(lam) if lambda_trainable else lam
        self.lambda_trainable = lambda_trainable

        # Norm & Dropout & storage for loss
        self.ln = nn.LayerNorm(dim_in) if use_layernorm else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # buffers for logging
        self.register_buffer("_last_sparsity", torch.tensor(0.0), persistent=False)
        self.register_buffer("_last_active_ratio", torch.tensor(0.0), persistent=False)

        # per-batch losses
        self.sparsity_loss = torch.tensor(0.0)
        self.balance_loss = torch.tensor(0.0)

    # ---- Optional NUFFT (1D 简化版：显式相位计算；若需真正 NUFFT，可替换为库调用) ----
    def _spectral_resp_nufft_1d(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        # 条件：至少一维可视作坐标
        if x.shape[-1] < 1:
            return None
        # 这里用显式相位 2πk·coord 作为 type-2 的轻量实现；当需要真正 NUFFT，可替换为 torchkbnufft 接口
        coord = x[:, :1]  # (B,1)
        k = self.Omega[0, :]  # (F,)
        with torch.no_grad():
            freq_norm = torch.abs(k)
        w = _es_window(freq_norm, fmax=self.es_fmax, beta=self.es_beta) * self.freq_gate  # (F,)
        z = 2 * math.pi * coord @ k[None, :]  # (B,F)
        cos_z = torch.cos(z) * w
        sin_z = torch.sin(z) * w
        spec_feat = torch.cat([cos_z, sin_z], dim=-1)  # (B, 2F)
        # 通道门控
        spec_feat = spec_feat * self.spec_gate
        h_spec = self.proj_spec(spec_feat)             # (B, D_out)
        return h_spec

    def _spectral_resp(self, x: torch.Tensor) -> torch.Tensor:
        # 若后端可用且形态满足，先走“NUFFT 风格”路径（这里是轻量实现）
        if self.nufft_backend is not None and x.shape[-1] >= 1:
            out = self._spectral_resp_nufft_1d(x)
            if out is not None:
                return out

        # ---- fallback: learnable RFF-style ----
        z = x @ self.Omega + self.Phi           # (B,F)
        with torch.no_grad():
            freq_norm = torch.norm(self.Omega, dim=0)
        w = _es_window(freq_norm, fmax=self.es_fmax, beta=self.es_beta) * self.freq_gate
        cos_z = torch.cos(z) * w
        sin_z = torch.sin(z) * w
        spec_feat = torch.cat([cos_z, sin_z], dim=-1)  # (B, 2F)
        # 通道门控
        spec_feat = spec_feat * self.spec_gate
        h_spec = self.proj_spec(spec_feat)             # (B, D_out)
        return h_spec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D_in)
        x_n = self.ln(x)

        # Local path
        h_local = self.base_act(self.local(x_n))  # (B, D_out)

        # Spectral path
        h_spec_raw = self._spectral_resp(x_n)     # (B, D_out)

        # Soft-thresholding
        lam = self.lam if isinstance(self.lam, torch.Tensor) else torch.tensor(self.lam, device=x.device)
        h_spec = soft_threshold(h_spec_raw, lam)

        # Sparsity stats & losses
        with torch.no_grad():
            # 统计响应级非零比例（作为 active_ratio 的 proxy）
            nz = (h_spec.abs() > 0).float().mean()
            self._last_sparsity = 1.0 - nz
            self._last_active_ratio = nz

        # L1：响应 + 通道门（与“有效 S”一致）
        resp_l1 = h_spec.abs().mean()
        gate_l1 = self.spec_gate.abs().mean()
        self.sparsity_loss = self.l1_alpha * (resp_l1 + 0.5 * gate_l1)

        # 分支能量平衡
        e_local = h_local.pow(2).mean()
        e_spec  = h_spec.pow(2).mean()
        self.balance_loss = (e_local - e_spec).pow(2)

        # Fuse
        y = h_local + h_spec
        y = self.drop(y)
        return y

    def stats(self) -> Dict[str, Any]:
        lam_val = float(self.lam.detach().item()) if isinstance(self.lam, torch.Tensor) else float(self.lam)
        return {
            "lambda": lam_val,
            "sparsity": float(self._last_sparsity.detach().item()),
            "active_ratio": float(self._last_active_ratio.detach().item()),
        }

    def extra_losses(self) -> Dict[str, torch.Tensor]:
        return {"l1_spectral": self.sparsity_loss, "balance": self.balance_loss}

# -----------------------------
# LibraKAN Block (layer + proj)
# -----------------------------
class LibraKANBlockExt(nn.Module):
    """
    One LibraKAN block:
       y = LinearGELU + Spectral (soft-threshold)  -> Linear projection to dim_out
    """
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        *,
        F: int = 128,
        spectral_scale: float = 1.0,
        es_beta: float = 8.0,
        es_fmax: Optional[float] = None,
        lambda_init: float = 0.01,
        lambda_trainable: bool = True,
        l1_alpha: float = 1e-4,
        base_activation: str = "gelu",
        use_layernorm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layer = LibraKANLayer(
            dim_in=dim_in,
            dim_out=dim_in,  # 内部保持等宽，再投影
            F=F,
            spectral_scale=spectral_scale,
            es_beta=es_beta,
            es_fmax=es_fmax,
            lambda_init=lambda_init,
            lambda_trainable=lambda_trainable,
            l1_alpha=l1_alpha,
            base_activation=base_activation,
            use_layernorm=use_layernorm,
            dropout=dropout,
        )
        self.proj = nn.Linear(dim_in, dim_out, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer(x)
        x = self.proj(x)
        return x

    @property
    def sparsity_loss(self) -> torch.Tensor:
        return self.layer.sparsity_loss

    @property
    def balance_loss(self) -> torch.Tensor:
        return self.layer.balance_loss

    def stats(self) -> Dict[str, Any]:
        return self.layer.stats()

    def extra_losses(self) -> Dict[str, torch.Tensor]:
        return self.layer.extra_losses()

class LibraKANMixerExt(nn.Module):
    """(N, dim) -> (N, dim) 的 mixer，用于替换 ResNet18/ViT/MLP-Mixer 等中的 MLP。"""
    def __init__(self, dim: int,
                 F: int = 128,
                 spectral_scale: float = 1.0,
                 es_beta: float = 8.0,
                 es_fmax: Optional[float] = None,
                 lambda_init: float = 0.01,
                 lambda_trainable: bool = True,
                 l1_alpha: float = 1.0,  # 建议设 1.0，全局权重在训练脚本外控
                 base_activation: str = "gelu",
                 use_layernorm: bool = False,
                 dropout: float = 0.0):
        super().__init__()
        self.block = LibraKANBlockExt(
            dim_in=dim, dim_out=dim,
            F=F, spectral_scale=spectral_scale,
            es_beta=es_beta, es_fmax=es_fmax,
            lambda_init=lambda_init, lambda_trainable=lambda_trainable,
            l1_alpha=l1_alpha, base_activation=base_activation,
            use_layernorm=use_layernorm, dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

    @property
    def sparsity_loss(self) -> torch.Tensor:
        return self.block.sparsity_loss

    @property
    def balance_loss(self) -> torch.Tensor:
        return self.block.balance_loss

    def extra_losses(self) -> Dict[str, torch.Tensor]:
        return self.block.extra_losses()

    def stats(self):
        return self.block.stats()

# -----------------------------
# LibraKAN Classifier
# -----------------------------
class LibraKANClassifierExt(nn.Module):
    """
    通用 LibraKAN 分类器（与 MLP/KAF/KAN/KAT 同接口）：
      - 输入 (B,D) 或 (B,C,H,W) 自动 flatten
      - 堆叠 LibraKANBlockExt，最后线性到 num_classes
      - 训练时通过 .extra_losses() 取到 L1 稀疏正则与 balance 正则
    """
    def __init__(
        self,
        input_shape: Union[Tuple[int], Sequence[int]],
        num_classes: int,
        hidden_dims: Sequence[int] = (256, 256),
        *,
        F: int = 128,
        spectral_scale: float = 1.0,
        es_beta: float = 8.0,
        es_fmax: Optional[float] = None,
        lambda_init: float = 0.01,
        lambda_trainable: bool = True,
        l1_alpha: float = 1e-4,
        base_activation: str = "gelu",
        use_layernorm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        if len(input_shape) == 1:
            D_in = input_shape[0]
        elif len(input_shape) == 3:
            C, H, W = input_shape
            D_in = C * H * W
        else:
            raise ValueError(f"Unsupported input_shape: {input_shape}")

        dims = [D_in] + list(hidden_dims)
        blocks = []
        for i in range(len(dims) - 1):
            blocks.append(
                LibraKANBlockExt(
                    dim_in=dims[i],
                    dim_out=dims[i + 1],
                    F=F,
                    spectral_scale=spectral_scale,
                    es_beta=es_beta,
                    es_fmax=es_fmax,
                    lambda_init=lambda_init,
                    lambda_trainable=lambda_trainable,
                    l1_alpha=l1_alpha,
                    base_activation=base_activation,
                    use_layernorm=use_layernorm,
                    dropout=dropout,
                )
            )
        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = torch.flatten(x, 1)
        elif x.dim() == 2:
            pass
        else:
            raise ValueError(f"Unsupported input tensor shape: {tuple(x.shape)}")
        x = self.backbone(x)
        x = self.head(x)
        return x

    def extra_losses(self) -> Dict[str, torch.Tensor]:
        # 聚合所有 block 的额外损失
        dev = _device_of(self.head.weight)
        l1_total = torch.zeros((), device=dev)
        bal_total = torch.zeros((), device=dev)
        for m in self.backbone.modules():
            if isinstance(m, LibraKANBlockExt):
                l1_total = l1_total + m.sparsity_loss.to(dev)
                bal_total = bal_total + m.balance_loss.to(dev)
        return {"l1_spectral": l1_total, "balance": bal_total}

    def stats(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for i, m in enumerate(self.backbone):
            if isinstance(m, LibraKANBlockExt):
                out[f"block{i}"] = m.stats()
        return out

# -----------------------------
# Thin adapter (兼容旧接口)
# -----------------------------
class LibraKAN(nn.Module):
    """
    Thin adapter for ResNet-18/ViT/Mixer head replacement:
      expects x in shape (B, dim) and returns (B, dim).
      Internally uses LibraKANBlockExt(dim_in=dim, dim_out=dim).
    """
    def __init__(self, dim: int, **kwargs):
        super().__init__()
        pass_keys = {
            "F", "spectral_scale", "es_beta", "es_fmax",
            "lambda_init", "lambda_trainable", "l1_alpha",
            "base_activation", "use_layernorm", "dropout"
        }
        filtered = {k: v for k, v in kwargs.items() if k in pass_keys}
        self.block = LibraKANBlockExt(dim_in=dim, dim_out=dim, **filtered)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

    @property
    def sparsity_loss(self) -> torch.Tensor:
        return self.block.sparsity_loss

    @property
    def balance_loss(self) -> torch.Tensor:
        return self.block.balance_loss

    def extra_losses(self) -> Dict[str, torch.Tensor]:
        return self.block.extra_losses()
