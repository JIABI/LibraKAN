import torch
import torch.nn as nn
import torch.nn.functional as F

class KAFBlock(nn.Module):
    def __init__(self, width: int, F: int = None, scale: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.width = width
        self.F = F or min(2*width, 2048)
        self.scale = scale
        self.dropout = nn.Dropout(dropout)

        freq = torch.randn(self.F, width) * self.scale
        self.register_buffer("freq", freq)

        self.alpha = nn.Parameter(torch.randn(self.F, width) * 0.02)
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        x = self.norm(x)
        proj = x.unsqueeze(1) * self.freq  # (N,F,W)
        sinf = torch.sin(proj)
        cosf = torch.cos(proj)
        feat = sinf + cosf
        feat = self.dropout(feat)
        h = (feat * self.alpha.unsqueeze(0)).sum(dim=1)
        return h

class MixerFFN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, F: int = None, scale: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden)
        self.kaf = KAFBlock(hidden, F=F, scale=scale, dropout=dropout)
        self.act = nn.GELU()
        self.proj_out = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = self.proj_in(x)
        x = self.kaf(x)
        x = self.act(x)
        x = self.proj_out(x)
        return x
