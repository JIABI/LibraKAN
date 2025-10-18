from typing import Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# 1) 解析官方 GPKAN/KAT 里用到的两类组件：
#    - KAT_Group：“rational”/可学习激活组（官方 CUDA 实现）
#    - KAN：两层 MLP（fc1+act+fc2），act 用 KAT_Group，等价“GPKAN 的 MLP 块”
#    下面的导入尽量兼容你贴的官方结构；若失败，请按你的 repo 路径手动改一下导入。
# ---------------------------------------------------------------------

def _resolve_kat_group_and_kan():
    KAT_Group, KAN = None, None
    err = []

    # Case A: 你把示例里那份文件直接变成模块（同目录/你的包里）
    import sys
    sys.path.insert(0, 'rational_kat_cu')
    try:
        from rational_kat_cu.kat_rational import KAT_Group as _KAT_Group  # e.g., rational_kat_cu/kat_rational.py 编译产物
        KAT_Group = _KAT_Group
    except Exception as e:
        err.append(("kat_rational.KAT_Group", e))

    # KAN 类（两层 MLP，act=KAT_Group）
    try:
        # 如果你把那份大文件命名为 kat_model.py（或保持原名），这里改成你的文件名
        # from kat_model import KAN as _KAN
        # KAN = _KAN
        # 为了更通用：有些人会直接把类放在 kat/ 下：
        from importlib import import_module
        for mod_name in ["kat_model", "kat", "models.kat_model", "models.kat"]:
            try:
                mod = import_module(mod_name)
                if hasattr(mod, "KAN"):
                    KAN = getattr(mod, "KAN")
                    break
            except Exception as e2:
                err.append((mod_name, e2))
    except Exception as e:
        err.append(("KAN import", e))

    # 如果 KAT_Group 未成功（例如你不想用 CUDA 版），给一个轻量 fallback（用 GELU 近似）
    if KAT_Group is None:
        class _KAT_Group_Fallback(nn.Module):
            def __init__(self, mode: str = "gelu", device=None):
                super().__init__()
                mode = (mode or "gelu").lower()
                if mode in ["swish", "silu"]:
                    self.act = nn.SiLU()
                elif mode in ["gelu"]:
                    self.act = nn.GELU()
                elif mode in ["identity", "linear"]:
                    self.act = nn.Identity()
                else:
                    # 其它字符串也先退化成 GELU
                    self.act = nn.GELU()

            def forward(self, x):
                return self.act(x)

        KAT_Group = _KAT_Group_Fallback

    # 如果 KAN 未成功导入，就用“官方结构”的最小复刻（两层 MLP + KAT_Group 激活）
    if KAN is None:
        class _KAN_Fallback(nn.Module):
            """
            与官方 KAN 类似结构：
                x = KAT_Group(mode="identity")(x) -> Dropout -> fc1
                x = KAT_Group(mode=act_init)(x)   -> Dropout -> fc2
            """

            def __init__(
                    self,
                    in_features,
                    hidden_features=None,
                    out_features=None,
                    act_layer=KAT_Group,
                    norm_layer=None,
                    bias=True,
                    drop=0.,
                    use_conv=False,
                    act_init="gelu",
                    device=None
            ):
                super().__init__()
                out_features = out_features or in_features
                hidden_features = hidden_features or in_features
                bias1 = True if isinstance(bias, bool) else bias[0]
                bias2 = True if isinstance(bias, bool) else bias[-1]

                self.fc1 = nn.Linear(in_features, hidden_features, bias=bias1)
                self.act1 = act_layer(mode="identity", device=device)
                self.drop1 = nn.Dropout(drop)

                self.norm = nn.Identity() if norm_layer is None else norm_layer(hidden_features)

                self.act2 = act_layer(mode=act_init, device=device)
                self.fc2 = nn.Linear(hidden_features, out_features, bias=bias2)
                self.drop2 = nn.Dropout(drop)

            def forward(self, x):
                x = self.act1(x)
                x = self.drop1(x)
                x = self.fc1(x)
                x = self.act2(x)
                x = self.drop2(x)
                x = self.fc2(x)
                return x

        KAN = _KAN_Fallback

    return KAT_Group, KAN, err


_KAT_Group, _KAN, _IMPORT_LOG = _resolve_kat_group_and_kan()


# ---------------------------------------------------------------------
# 2) GPKAN Block：对齐你之前的 API（dim_in -> dim_out），内部用官方 KAN MLP
# ---------------------------------------------------------------------
class GPKANBlockExt(nn.Module):
    """
    一层 GPKAN（官方 KAT 风格的 KAN MLP）：
        y = KAN(in_features=dim_in, hidden=dim_hidden, out=dim_out, act_layer=KAT_Group, act_init=...)
    """

    def __init__(
            self,
            dim_in: int,
            dim_out: int,
            *,
            mlp_ratio: float = 4.0,
            drop: float = 0.0,
            act_init: str = "gelu",  # "gelu" / "swish"（对应官方代码）
            use_norm: bool = False,  # 官方类里有 norm 位，但默认 Identity
            bias: bool = True,
            device: str = None,
    ):
        super().__init__()
        hidden = int(dim_in * mlp_ratio)
        norm_layer = (nn.LayerNorm if use_norm else None)
        self.gpkan = _KAN(
            in_features=dim_in,
            hidden_features=hidden,
            out_features=dim_out,
            act_layer=_KAT_Group,  # 官方 KAT_Group（或 fallback）
            norm_layer=norm_layer,
            bias=bias,
            drop=drop,
            use_conv=False,  # 小型分类器用 Linear
            act_init=act_init,  # "gelu"/"swish"
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gpkan(x)


# ---------------------------------------------------------------------
# 3) GPKAN 分类器：与 KAF/MLP/KAN 保持一致
#    - 输入 (B,D) 或 (B,C,H,W) 自动 flatten
#    - 堆叠若干 GPKANBlockExt，最后线性到 num_classes
# ---------------------------------------------------------------------
class GPKANClassifierExt(nn.Module):
    """
    通用 GPKAN 分类器（与 KAF/MLP/KAN 封装同接口）：
      - 支持 1D/3D 输入（自动 flatten）
      - 层内为“官方 KAN MLP + KAT_Group 激活”的结构
    """

    def __init__(
            self,
            input_shape: Union[Tuple[int], Sequence[int]],
            num_classes: int,
            hidden_dims: Sequence[int] = (256, 256),
            *,
            mlp_ratio: float = 4.0,
            drop: float = 0.0,
            act_init: str = "gelu",  # "gelu" or "swish"
            use_norm: bool = False,
            bias: bool = True,
            device: str = None,
    ):
        super().__init__()
        # 解析输入维度
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
                GPKANBlockExt(
                    dim_in=dims[i],
                    dim_out=dims[i + 1],
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    act_init=act_init,
                    use_norm=use_norm,
                    bias=bias,
                    device=device,
                )
            )
        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 接受 (B,D) 或 (B,C,H,W)
        if x.dim() == 4:
            x = torch.flatten(x, 1)
        elif x.dim() == 2:
            pass
        else:
            raise ValueError(f"Unsupported input tensor shape: {tuple(x.shape)}")
        x = self.backbone(x)
        x = self.head(x)
        return x