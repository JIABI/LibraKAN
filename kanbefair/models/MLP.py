# models/mlp_ext.py

from typing import Sequence, Tuple, Union

import torch

import torch.nn as nn

import torch.nn.functional as F


def _act_module(name: str) -> nn.Module:
    name = name.lower()

    if name == "gelu":
        return nn.GELU()

    if name == "relu":
        return nn.ReLU(inplace=True)

    if name in ["silu", "swish"]:
        return nn.SiLU(inplace=True)

    raise ValueError(f"Unsupported activation: {name}")


class MLPBlock(nn.Module):
    """

    One MLP block: (optional LayerNorm) -> Linear -> Activation -> (optional Dropout)

    Args:

        dim_in:  input dimension

        dim_out: output dimension

        activation: "gelu" | "relu" | "silu"

        use_layernorm: whether to apply LayerNorm on input

        dropout: dropout prob after activation

    """

    def __init__(

            self,

            dim_in: int,

            dim_out: int,

            activation: str = "gelu",

            use_layernorm: bool = False,

            dropout: float = 0.0,

    ):
        super().__init__()

        self.ln = nn.LayerNorm(dim_in) if use_layernorm else nn.Identity()

        self.fc = nn.Linear(dim_in, dim_out, bias=True)

        self.act = _act_module(activation)

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)

        x = self.fc(x)

        x = self.act(x)

        x = self.drop(x)

        return x


class MLPClassifierExt(nn.Module):
    """

    Generic MLP classifier with stacked MLPBlocks and a final linear head.

    Supports:

      - vector input: (B, D)  when input_shape=(D,)

      - image input:  (B, C, H, W)  when input_shape=(C,H,W); it will be flattened to D=C*H*W

    Example:

        # MNIST: 1x28x28 -> 10 classes

        model = MLPClassifierExt(input_shape=(1,28,28), num_classes=10,

                                 hidden_dims=[256,256], activation="gelu")

        # CIFAR-10: 3x32x32 -> 10 classes

        model = MLPClassifierExt(input_shape=(3,32,32), num_classes=10,

                                 hidden_dims=[512], activation="relu")

    """

    def __init__(

            self,

            input_shape: Union[Tuple[int], Sequence[int]],

            num_classes: int,

            hidden_dims: Sequence[int] = (256, 256),

            *,

            activation: str = "gelu",

            use_layernorm: bool = False,

            dropout: float = 0.0,

    ):

        super().__init__()

        # resolve input dim

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

                MLPBlock(

                    dim_in=dims[i],

                    dim_out=dims[i + 1],

                    activation=activation,

                    use_layernorm=use_layernorm,

                    dropout=dropout,

                )

            )

        self.backbone = nn.Sequential(*blocks)

        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # accept (B,D) or (B,C,H,W)

        if x.dim() == 4:

            x = torch.flatten(x, 1)

        elif x.dim() == 2:

            pass

        else:

            raise ValueError(f"Unsupported input tensor shape: {tuple(x.shape)}")

        x = self.backbone(x)

        x = self.head(x)

        return x

