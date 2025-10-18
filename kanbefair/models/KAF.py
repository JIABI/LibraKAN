from typing import Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from kaf_act import RFFActivation

def _to_tuple(x):
    return tuple(x) if isinstance(x, (list, tuple)) else (x,)


class KAFBlockExt(nn.Module):
    """
    一层 KAF block = 先做 RFFActivation，再线性映射到下一个维度。
    维度约定：RFFActivation 的输入输出都是 dim_in（通道数不变），
    再用线性层把维度从 dim_in -> dim_out。
    """

    def __init__(
            self,
            dim_in: int,
            dim_out: int,
            # RFFActivation 的关键超参（与你贴的示例一致）
            num_grids: int = 16,
            activation_expectation: float = 1.64,
            base_activation=F.gelu,  # 也可传 F.silu
            use_layernorm: bool = False,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.kaf = RFFActivation(
            num_grids=num_grids,
            activation_expectation=activation_expectation,
            base_activation=base_activation,
            use_layernorm=use_layernorm,
            dropout=dropout,
        )
        # 注意：大多数实现默认 RFFActivation 不改变通道数，这里线性层负责变换维度
        self.proj = nn.Linear(dim_in, dim_out, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 期望输入 (B, D)
        x = self.kaf(x)
        x = self.proj(x)
        return x


class KAFClassifierExt(nn.Module):
    """
    通用 KAF 分类器：
      - 输入可为 1D: (B, D)
      - 或 3D 图像: (B, C, H, W) —— 会先 flatten 成 (B, C*H*W)
      - 堆叠若干 KAFBlockExt，最后线性到 num_classes

    参数：
      input_shape: (D,) 或 (C,H,W)
      hidden_dims: 如 [256, 256]
      num_grids: RFFActivation 的频率网格数（论文 Figure 2 常用 9/16 等）
      base_activation: F.gelu 或 F.silu
    """

    def __init__(
            self,
            input_shape: Union[Tuple[int], Sequence[int]],
            num_classes: int,
            hidden_dims: Sequence[int] = (256, 256),
            *,
            num_grids: int = 16,
            activation_expectation: float = 1.64,
            base_activation=F.gelu,
            use_layernorm: bool = False,
            dropout: float = 0.0,
    ):
        super().__init__()
        shp = _to_tuple(input_shape)
        if len(shp) == 1:
            D_in = shp[0]
        elif len(shp) == 3:
            C, H, W = shp
            D_in = C * H * W
        else:
            raise ValueError(f"Unsupported input_shape: {input_shape}")

        dims = [D_in] + list(hidden_dims)
        blocks = []
        for i in range(len(dims) - 1):
            blocks.append(
                KAFBlockExt(
                    dim_in=dims[i],
                    dim_out=dims[i + 1],
                    num_grids=num_grids,
                    activation_expectation=activation_expectation,
                    base_activation=base_activation,
                    use_layernorm=use_layernorm,
                    dropout=dropout,
                )
            )
        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Linear(dims[-1], num_classes)
        self._expects_image = (len(shp) == 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 接受 (B, D) 或 (B, C, H, W)
        if x.dim() == 4:
            x = torch.flatten(x, 1)
        elif x.dim() == 2:
            pass
        else:
            raise ValueError(f"Unsupported input tensor shape: {tuple(x.shape)}")
        x = self.backbone(x)
        x = self.head(x)
        return x


# ---- KAF adapter: wrap to (N, dim)->(N, dim), auto-map ctor names ----

class KAFMixerExt(nn.Module):
    """

    适配器：把 KAF 封装为 (N, dim)->(N, dim) 的 mixer，

    方便在 ResNet18 的 MLP 位置无缝替换。

    """

    def __init__(self, dim: int,

                 num_grids: int = 16,

                 activation_expectation: float = 1.64,

                 base_activation=F.gelu,

                 use_layernorm: bool = False,

                 dropout: float = 0.0):
        super().__init__()

        # 用一层 KAFBlockExt，并保持 in/out 维度一致

        self.block = KAFBlockExt(

            dim_in=dim,

            dim_out=dim,

            num_grids=num_grids,

            activation_expectation=activation_expectation,

            base_activation=base_activation,

            use_layernorm=use_layernorm,

            dropout=dropout,

        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# 可选：提供一个工厂函数，训练脚本也能用函数名加载

def build_kaf_mixer(dim: int, **kwargs) -> nn.Module:
    return KAFMixerExt(dim=dim, **kwargs)

