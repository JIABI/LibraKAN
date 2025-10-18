# models/fan_ext.py

from typing import Sequence, Tuple, Union

import math

import torch

import torch.nn as nn

import torch.nn.functional as F


def _act_module(name: str) -> nn.Module:
    name = name.lower()

    if name == "gelu":
        return nn.GELU()

    if name in ["silu", "swish"]:
        return nn.SiLU(inplace=True)

    if name == "relu":
        return nn.ReLU(inplace=True)

    if name in ["identity", "linear"]:
        return nn.Identity()

    raise ValueError(f"Unsupported activation: {name}")


class FANLayer(nn.Module):
    """

    FAN 单层，实现自官方 README 的公式：

        φ(x) = [cos(W_p x) || sin(W_p x) || σ(B_{pbar} + W_{pbar} x)]

    设计要点：

      - 输出维度 = d_out

      - 其中 p_ratio * d_out（向下取偶数）用于 Fourier 分支，剩余用于 base 分支

      - Fourier 分支先计算 z = x W_f + b_f, 然后 concat [cos(z), sin(z)]

      - Base 分支做线性 + 非线性：σ(x W_b + b_b)

    参数：

      dim_in: 输入维度 d_in

      dim_out: 输出维度 d_out

      p_ratio: Fourier 通道占比（官方示例 p=0.25）

      base_activation: base 分支的激活（"gelu"/"silu"/"relu"...）

      use_layernorm: layernorm 到输入

      dropout: 输出处的 dropout

      w_init_scale: 频率矩阵初始化尺度（控制频率大小）

    """

    def __init__(

            self,

            dim_in: int,

            dim_out: int,

            *,

            p_ratio: float = 0.25,

            base_activation: str = "gelu",

            use_layernorm: bool = False,

            dropout: float = 0.0,

            w_init_scale: float = 1.0,

    ):

        super().__init__()

        assert 0.0 <= p_ratio < 1.0, "p_ratio must be in [0,1)"

        self.dim_in = dim_in

        self.dim_out = dim_out

        self.p_ratio = p_ratio

        # 为了让 2M = d_p 成立（cos+sin），把 d_p 取为偶数

        d_p = int(math.floor(dim_out * p_ratio))

        if d_p % 2 == 1:
            d_p -= 1

        self.d_p = max(0, min(dim_out, d_p))  # Fourier 通道数（总）

        self.M = self.d_p // 2  # 频率个数（每个 freq 产出 cos 与 sin 两维）

        self.d_base = dim_out - self.d_p  # base 分支通道数

        self.act = _act_module(base_activation)

        self.ln = nn.LayerNorm(dim_in) if use_layernorm else nn.Identity()

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # -------- Fourier 分支参数：W_f ∈ R^{d_in × M}, b_f ∈ R^{M}

        if self.M > 0:

            self.W_f = nn.Parameter(torch.randn(dim_in, self.M) * (w_init_scale / math.sqrt(dim_in)))

            self.b_f = nn.Parameter(torch.empty(self.M).uniform_(0.0, 2.0 * math.pi))

        else:

            self.register_parameter("W_f", None)

            self.register_parameter("b_f", None)

        # -------- Base 分支参数：W_b ∈ R^{d_in × d_base}, b_b ∈ R^{d_base}

        if self.d_base > 0:

            self.W_b = nn.Parameter(torch.randn(dim_in, self.d_base) * (1.0 / math.sqrt(dim_in)))

            self.b_b = nn.Parameter(torch.zeros(self.d_base))

        else:

            self.register_parameter("W_b", None)

            self.register_parameter("b_b", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x: (B, d_in)

        x = self.ln(x)

        outs = []

        # Fourier 分支

        if self.M > 0:
            z = x @ self.W_f + self.b_f  # (B, M)

            fourier = torch.cat([torch.cos(z), torch.sin(z)], dim=-1)  # (B, 2M) == (B, d_p)

            outs.append(fourier)

        # Base 分支

        if self.d_base > 0:
            base = x @ self.W_b + self.b_b  # (B, d_base)

            base = self.act(base)

            outs.append(base)

        if len(outs) == 1:

            y = outs[0]

        else:

            y = torch.cat(outs, dim=-1)  # (B, d_out)

        y = self.drop(y)

        return y


class FANBlockExt(nn.Module):
    """

    与其他模型保持一致的 Block：先 FANLayer (保持输出维度 dim_in)，再线性映射到 dim_out

    """

    def __init__(

            self,

            dim_in: int,

            dim_out: int,

            *,

            p_ratio: float = 0.25,

            base_activation: str = "gelu",

            use_layernorm: bool = False,

            dropout: float = 0.0,

            w_init_scale: float = 1.0,

    ):
        super().__init__()

        self.fan = FANLayer(

            dim_in=dim_in,

            dim_out=dim_in,  # 先做逐通道 FAN 非线性（不改通道数）

            p_ratio=p_ratio,

            base_activation=base_activation,

            use_layernorm=use_layernorm,

            dropout=dropout,

            w_init_scale=w_init_scale,

        )

        self.proj = nn.Linear(dim_in, dim_out, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.fan(x))


class FANClassifierExt(nn.Module):
    """

    通用 FAN 分类器（与 MLP/KAF/KAN/GPKAN 保持统一接口）：

      - 输入 (B, D) 或 (B, C, H, W) 自动 flatten

      - 堆叠 FANBlockExt，最后线性到 num_classes

    """

    def __init__(

            self,

            input_shape: Union[Tuple[int], Sequence[int]],

            num_classes: int,

            hidden_dims: Sequence[int] = (256, 256),

            *,

            p_ratio: float = 0.25,

            base_activation: str = "gelu",

            use_layernorm: bool = False,

            dropout: float = 0.0,

            w_init_scale: float = 1.0,

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

                FANBlockExt(

                    dim_in=dims[i],

                    dim_out=dims[i + 1],

                    p_ratio=p_ratio,

                    base_activation=base_activation,

                    use_layernorm=use_layernorm,

                    dropout=dropout,

                    w_init_scale=w_init_scale,

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

