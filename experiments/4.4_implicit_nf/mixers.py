import torch
import torch.nn as nn
from librakan_act import make_librakan_mixer

def make_mlp(dim, hidden_dim):
    return nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))

def make_kaf_adapter(dim, hidden_dim):
    try:
        from kaf_act import RFFActivation
        import torch.nn.functional as F
    except Exception as e:
        raise ImportError("kaf_act not found. Install: pip install git+https://github.com/kolmogorovArnoldFourierNetwork/KAF.git") from e
    class KAFBlock(nn.Module):
        def __init__(self, dim, hidden):
            super().__init__()
            self.fc1 = nn.Linear(dim, hidden)
            self.kaf = RFFActivation(base_activation=F.silu)
            self.fc2 = nn.Linear(hidden, dim)
        def forward(self, x):
            return self.fc2(self.kaf(self.fc1(x)))
    return KAFBlock(dim, hidden_dim)

def make_mixer(name, dim, hidden_dim, **libra_kwargs):
    name = name.lower()
    if name == "mlp":
        return make_mlp(dim, hidden_dim)
    if name == "kaf":
        return make_kaf_adapter(dim, hidden_dim)
    if name == "librakan":
        return make_librakan_mixer(dim, hidden_dim, **libra_kwargs)
    raise ValueError(f"Unknown mixer for 4.4: {name}. Choose from mlp/kaf/librakan")
