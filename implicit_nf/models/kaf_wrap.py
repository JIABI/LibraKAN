
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from kaf_act import RFFActivation
    HAS_RFF = True
except Exception:
    HAS_RFF = False

class KAFBlock(nn.Module):
    def __init__(self, dim_in, dim_out, F=256):
        super().__init__()
        if not HAS_RFF:
            raise ImportError("kaf_act.RFFActivation not found. Please install/put it in PYTHONPATH.")
        self.lin = nn.Linear(dim_in, dim_out, bias=True)
        self.rff = RFFActivation()  # uses official KAF activation

    def forward(self, x):
        return self.rff(self.lin(x))

class KAFImplicit(nn.Module):
    def __init__(self, in_dim=2, hidden=256, depth=8, out_dim=3, F=512):
        super().__init__()
        if not HAS_RFF:
            raise ImportError("kaf_act.RFFActivation not found.")
        layers = []
        D = in_dim
        for i in range(depth):
            layers += [nn.Linear(D, hidden, bias=True), RFFActivation()]
            D = hidden
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, out_dim, bias=True)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        h = self.backbone(coords)
        y = self.head(h)
        y = torch.sigmoid(y)
        return y

    def active_freq(self, tau: float = 1e-3) -> int:
        total = 0
        for m in self.modules():
            if isinstance(m, RFFActivation):
                if hasattr(m, "active_freq"):
                    total += int(m.active_freq(tau))
                else:
                    if getattr(m, "rff", None) is not None and hasattr(m.rff, "combination"):
                        W = m.rff.combination.weight.detach()  # [H, 2G]
                        per_feat = W.abs().mean(dim=0)  # (2G,)
                        G = per_feat.numel() // 2
                        per_freq = torch.stack([per_feat[:G], per_feat[G:]], dim=0).max(dim=0).values
                        total += int((per_freq > tau).sum().item())
        return total