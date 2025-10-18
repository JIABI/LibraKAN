import torch
import torch.nn as nn

class MLPBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(width, width)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class MixerFFN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden)
        self.mixer = MLPBlock(hidden)
        self.act = nn.GELU()
        self.proj_out = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = self.proj_in(x)
        x = self.mixer(x)
        x = self.act(x)
        x = self.proj_out(x)
        return x
