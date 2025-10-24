# models/kan.py  —— Periodic KAN (方案A)

import torch

import torch.nn as nn

from typing import Optional


# ---- cubic B-spline basis ----

def cubic_bspline(t: torch.Tensor) -> torch.Tensor:
    """

    t: (N,) in [0,1)

    return: (N,4) basis values

    """

    t0 = ((1 - t) ** 3) / 6.0

    t1 = (3 * t ** 3 - 6 * t ** 2 + 4) / 6.0

    t2 = (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6.0

    t3 = (t ** 3) / 6.0

    return torch.stack([t0, t1, t2, t3], dim=-1)


def _make_act(act: Optional[str]) -> nn.Module:
    if act is None or str(act).lower() in ("", "none", "id", "identity"):
        return nn.Identity()

    a = str(act).lower()

    if a == "relu": return nn.ReLU()

    if a == "silu": return nn.SiLU()

    return nn.GELU()  # default


class KAN1D(nn.Module):
    """

    Periodic 1D KAN (方案A):

      x --[normalize by fixed train range]--> u in R

         -> u_mod = u % K  (periodic wrap across K cells)

         -> cubic B-spline on fractional part in each cell

         -> features (N,K) with periodic indexing

         -> [LayerNorm] -> head

    - act=None 支持（纯 KAN：非线性仅由样条产生）

    - hidden=None:  Linear(K,1)

      hidden=int :  Linear(K,H) -> act -> Linear(H,1)

    - set_input_range(xmin,xmax) 用于与数据的 train_range 对齐

    """

    def __init__(self,

                 num_knots: int = 256,

                 hidden: Optional[int] = None,

                 train_xmin: float = -3.0,

                 train_xmax: float = 3.0,

                 act: Optional[str] = None,

                 use_layernorm: bool = True,

                 dtype: torch.dtype = torch.float32):

        super().__init__()

        assert num_knots >= 4

        self.num_knots = int(num_knots)

        self.hidden = None if hidden is None else int(hidden)

        # 固定训练范围（评估与外推沿用同一缩放）

        self.register_buffer("x_min_buf", torch.tensor(float(train_xmin), dtype=dtype))

        self.register_buffer("x_max_buf", torch.tensor(float(train_xmax), dtype=dtype))

        self.use_layernorm = use_layernorm

        if use_layernorm:
            self.norm = nn.LayerNorm(self.num_knots, elementwise_affine=True)

        if self.hidden is None:

            self.head = nn.Linear(self.num_knots, 1, bias=True, dtype=dtype)

            nn.init.xavier_uniform_(self.head.weight, gain=2.0)

            self.act = nn.Identity()

        else:

            self.feat2hid = nn.Linear(self.num_knots, self.hidden, bias=True, dtype=dtype)

            self.hid2out = nn.Linear(self.hidden, 1, bias=True, dtype=dtype)

            nn.init.xavier_uniform_(self.feat2hid.weight, gain=2.0)

            nn.init.xavier_uniform_(self.hid2out.weight, gain=2.0)

            self.act = _make_act(act)

            self.head = None

    @torch.no_grad()
    def set_input_range(self, xmin: float, xmax: float):

        """确保与 make_data(..., train_range=(xmin,xmax)) 使用同一范围"""

        self.x_min_buf.fill_(float(xmin))

        self.x_max_buf.fill_(float(xmax))

    # -------- periodic spline features --------

    def spline_features(self, x: torch.Tensor) -> torch.Tensor:

        """

        周期三次B样条：

          u = ((x - xmin)/(xmax - xmin)) * K

          u_mod = u % K                      # 周期回环

          i = floor(u_mod)                   # 当前cell起点

          frac = u_mod - i                   # ∈ [0,1)

          用 i..i+3 的权重（索引同样周期回环）scatter_add 到 K维特征

        """

        K = self.num_knots

        x = x.to(self.x_min_buf.dtype)

        # 固定训练期缩放

        x0 = (x - self.x_min_buf) / (self.x_max_buf - self.x_min_buf + 1e-8)  # (N,1)

        u = x0 * K  # (N,1)

        u = torch.remainder(u, K)  # (N,1) in [0,K)

        i = torch.floor(u).long()  # (N,1)

        frac = (u - i.to(u.dtype)).squeeze(-1)  # (N,), in [0,1)

        B = cubic_bspline(frac)  # (N,4)

        feats = torch.zeros(x.shape[0], K, device=x.device, dtype=x.dtype)  # (N,K)

        base = i.squeeze(-1)  # (N,)

        for j in range(4):
            idx = (base + j) % K  # 周期索引

            feats.scatter_add_(1, idx.unsqueeze(-1), B[:, j].unsqueeze(-1))

        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        feats = self.spline_features(x)  # (N,K)

        if self.use_layernorm:
            feats = self.norm(feats)

        if self.hidden is None:
            return self.head(feats)  # (N,1)

        h = self.feat2hid(feats)

        h = self.act(h)  # act=None -> Identity

        return self.hid2out(h)

