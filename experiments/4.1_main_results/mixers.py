
import torch
import torch.nn as nn
from typing import Optional
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

def make_kan_adapter(dim, hidden_dim):
    try:
        from kan.MultKAN import MultKAN
    except Exception as e:
        raise ImportError("pykan not found. Install: pip install git+https://github.com/KindXiaoming/pykan.git") from e
    class KANBlock(nn.Module):
        def __init__(self, dim, hidden):
            super().__init__()
            self.inp = nn.Linear(dim, hidden)
            self.kan = MultKAN(width=hidden, depth=1)
            self.out = nn.Linear(hidden, dim)
        def forward(self, x):
            h = torch.relu(self.inp(x))
            h = self.kan(h)
            return self.out(h)
    return KANBlock(dim, hidden_dim)

def make_kat_adapter(dim, hidden_dim):
    try:
        from katransformer import KAEncoderLayer
    except Exception as e:
        raise ImportError("KAT not found. Install: pip install git+https://github.com/Adamdad/kat.git and ensure katransformer.py in path.") from e
    class KATBlock(nn.Module):
        def __init__(self, dim, hidden):
            super().__init__()
            self.inp = nn.Linear(dim, hidden)
            self.mixer = KAEncoderLayer(d_model=hidden, nhead=4, dim_feedforward=hidden)
            self.out = nn.Linear(hidden, dim)
        def forward(self, x):
            h = torch.relu(self.inp(x))
            h = h.unsqueeze(1)
            h = self.mixer(h)
            h = h.squeeze(1)
            return self.out(h)
    return KATBlock(dim, hidden_dim)

def make_fan_adapter(dim, hidden_dim):
    # FAN: https://github.com/YihongDong/FAN/blob/main/FANLayer.py
    try:
        from FANLayer import FANLayer
    except Exception as e:
        raise ImportError("FANLayer not found. Place FANLayer.py in PYTHONPATH or install the FAN repo.") from e
    class FANBlock(nn.Module):
        def __init__(self, dim, hidden):
            super().__init__()
            self.inp = nn.Linear(dim, hidden)
            # The FANLayer expects (B, N, C); we will wrap a singleton token length.
            self.fan = FANLayer(dim=hidden)
            self.out = nn.Linear(hidden, dim)
        def forward(self, x):
            # x: (B,C) -> (B,1,C)
            h = torch.relu(self.inp(x))
            h = h.unsqueeze(1)
            h = self.fan(h)
            h = h.squeeze(1)
            return self.out(h)
    return FANBlock(dim, hidden_dim)

def make_mixer(name: str, dim: int, hidden_dim: int, **libra_kwargs) -> nn.Module:
    name = name.lower()
    if name == "mlp":
        return make_mlp(dim, hidden_dim)
    if name == "kaf":
        return make_kaf_adapter(dim, hidden_dim)
    if name == "kan":
        return make_kan_adapter(dim, hidden_dim)
    if name == "kat":
        return make_kat_adapter(dim, hidden_dim)
    if name == "fan":
        return make_fan_adapter(dim, hidden_dim)
    if name == "librakan":
        return make_librakan_mixer(dim, hidden_dim, **libra_kwargs)
    raise ValueError(f"Unknown mixer: {name}")
