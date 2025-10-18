import torch
import torch.nn as nn
from mixers import MixerFFN

class Classifier(nn.Module):
    def __init__(self, in_dim=784, width=128, depth=2, num_classes=10, mixer="librakan", **mkw):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(depth):
            layers += [MixerFFN(in_dim=dim, hidden=width, out_dim=width, kind=mixer, **mkw)]
            dim = width
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(width, num_classes)
    def forward(self, x):
        y = x.view(x.size(0), -1)
        y = self.trunk(y)
        return self.head(y)

class ImplicitReconstructor(nn.Module):
    def __init__(self, width=128, depth=4, channels=1, mixer="librakan", **mkw):
        super().__init__()
        layers = [nn.Linear(2, width), nn.GELU()]
        for _ in range(depth-1):
            layers += [MixerFFN(in_dim=width, hidden=width, out_dim=width, kind=mixer, **mkw)]
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, channels)
    def forward(self, coords):
        y = self.body(coords)
        y = self.head(y).clamp(0.0, 1.0)
        return y
