
import torch
import torch.nn as nn
import torch.nn.functional as F

def _act(name:str):
    name = name.lower()
    if name == "gelu": return nn.GELU()
    if name == "relu": return nn.ReLU(inplace=True)
    if name in ["silu","swish"]: return nn.SiLU(inplace=True)
    return nn.Identity()

class ImplicitMLP(nn.Module):
    """ [2 -> W] x L -> C  , no external PE """
    def __init__(self, in_dim=2, hidden=256, depth=8, out_dim=3, act="gelu"):
        super().__init__()
        layers = []
        D = in_dim
        for i in range(depth):
            layers += [nn.Linear(D, hidden, bias=True), _act(act)]
            D = hidden
        layers += [nn.Linear(D, out_dim, bias=True)]
        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (B,2) in [-1,1]^2 → (B,C) in [0,1]
        y = self.net(coords)
        y = torch.sigmoid(y)  # map to [0,1]
        return y
