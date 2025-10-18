import torch
import torch.nn as nn
from typing import Sequence, Union, Tuple


# ===========================================================
#  ✅ 1. Factory — 精确匹配你本地 pykan.kan.KANLayer
# ===========================================================
def _resolve_kan_linear_factory():
    """
    Factory for your installed pykan version.
    Compatible with: KANLayer(in_dim, out_dim, num, k, grid_range=[-R, R], ...)
    """
    try:
        from pykan.kan.KANLayer import KANLayer as _KANLayer
    except ImportError as e:
        raise ImportError(
            f"❌ Cannot import KANLayer from pykan.kan.KANLayer. "
            f"Please ensure pykan is installed. Original error: {e}"
        )

    def factory(in_f, out_f, grid, degree, x_range):
        return _KANLayer(
            in_dim=in_f,
            out_dim=out_f,
            num=grid,  # your version uses “num” instead of grid
            k=degree,  # spline order
            grid_range=[-float(x_range), float(x_range)],
            device="cuda" if torch.cuda.is_available() else "cpu",
            sparse_init=False,
            save_plot_data=False,
        )

    return factory


# ===========================================================
#  ✅ 2. Basic KAN Block
# ===========================================================
class KANPyBlock(nn.Module):

    def __init__(

            self,

            in_features: int,

            out_features: int,

            grid: int = 5,

            degree: int = 3,

            x_range: float = 1.0,

            use_layernorm: bool = False,

            p_drop: float = 0.0,

    ):
        super().__init__()

        make_kan = _resolve_kan_linear_factory()

        self.kan = make_kan(in_features, out_features, grid, degree, x_range)

        self.norm = nn.LayerNorm(out_features) if use_layernorm else nn.Identity()

        self.drop = nn.Dropout(p_drop) if p_drop > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 兼容 tuple 输出（pykan 返回 (out, spline_vals)）

        x = self.kan(x)

        if isinstance(x, tuple):
            x = x[0]  # 只保留主输出

        x = self.norm(x)

        x = self.drop(x)

        return x


# ===========================================================
#  ✅ 3. KAN Classifier
# ===========================================================
class KANPyClassifier(nn.Module):
    """
    Flatten -> multiple KAN blocks -> Linear head
    """

    def __init__(
            self,
            input_shape: Union[Tuple[int], Sequence[int]],
            num_classes: int,
            widths: Sequence[int],
            grid: int = 5,
            degree: int = 3,
            x_range: float = 1.0,
            use_layernorm: bool = False,
            p_drop: float = 0.0,
    ):
        super().__init__()

        # ---- Input flattening logic ----
        if isinstance(input_shape, (list, tuple)) and len(input_shape) > 1:
            in_dim = 1
            for d in input_shape:
                in_dim *= int(d)
            self._need_flatten = True
        else:
            in_dim = int(input_shape[0]) if isinstance(input_shape, (list, tuple)) else int(input_shape)
            self._need_flatten = False

        # ---- Backbone ----
        dims = [in_dim] + list(widths)
        blocks = []
        for i in range(len(dims) - 1):
            blocks.append(
                KANPyBlock(
                    dims[i],
                    dims[i + 1],
                    grid=grid,
                    degree=degree,
                    x_range=x_range,
                    use_layernorm=use_layernorm,
                    p_drop=p_drop,
                )
            )
        self.backbone = nn.Sequential(*blocks) if blocks else nn.Identity()

        # ---- Classification head ----
        self.head = nn.Linear(dims[-1], num_classes)

        # ---- Parameter stats ----
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._need_flatten and x.dim() > 2:
            x = torch.flatten(x, 1)
        x = self.backbone(x)
        x = self.head(x)
        return x


# ===========================================================
#  ✅ 4. Convenience Builder (for build_model(...) usage)
# ===========================================================
def build_kan_classifier(
        input_shape: Union[Tuple[int], Sequence[int]],
        num_classes: int,
        hidden: Union[int, Sequence[int]] = 256,  # ← 支持 int 或 list
        depth: int = 1,
        grid: int = 5,
        degree: int = 3,
        x_range: float = 1.0,
        use_layernorm: bool = False,
        p_drop: float = 0.0,
) -> nn.Module:
    # 规范化 widths
    if isinstance(hidden, int):
        widths = [hidden] * depth
    elif isinstance(hidden, (list, tuple)):
        if len(hidden) == 1:
            widths = [int(hidden[0])] * depth
        else:
            widths = [int(h) for h in hidden]
            # 如果用户传了多个 hidden，自动以其长度为 depth
            depth = len(widths)
    else:
        widths = [256] * depth

    model = KANPyClassifier(
        input_shape=input_shape,
        num_classes=num_classes,
        widths=widths,
        grid=grid,
        degree=degree,
        x_range=x_range,
        use_layernorm=use_layernorm,
        p_drop=p_drop,
    )
    return model