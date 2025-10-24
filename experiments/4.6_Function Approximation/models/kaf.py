
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RandomFourierFeatures(nn.Module):
    def __init__(self, input_dim: int, num_grids: int, dropout: float = 0.0, activation_expectation: float = 1.64):
        super().__init__()
        self.input_dim = input_dim
        self.num_grids = num_grids
        self.dropout = nn.Dropout(dropout)
        var_w = 1.0 / (input_dim * activation_expectation)
        self.weight = nn.Parameter(torch.randn(input_dim, num_grids) * math.sqrt(var_w))
        self.bias = nn.Parameter(torch.empty(num_grids))
        nn.init.uniform_(self.bias, 0, 2*math.pi)
        self.combination = nn.Linear(2*num_grids, input_dim)
        nn.init.xavier_uniform_(self.combination.weight)
        if self.combination.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.combination.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.combination.bias, -bound, bound)

    def forward(self, x):
        proj = x @ self.weight + self.bias
        ff = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        ff = self.dropout(ff)
        out = self.combination(ff)
        return out

class FastKAFLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_grids: int = 16, use_layernorm: bool = True,
                 spline_dropout: float = 0.0, base_activation = F.gelu, activation_expectation: float = 1.64):
        super().__init__()
        self.base_activation = base_activation
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layernorm = nn.LayerNorm(input_dim) if use_layernorm and input_dim > 1 else None
        self.feature_transform = RandomFourierFeatures(input_dim, num_grids, spline_dropout, activation_expectation)
        self.base_scale = nn.Parameter(torch.tensor(1.0))
        self.spline_scale = nn.Parameter(torch.tensor(1e-2))
        self.final_linear = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(self.final_linear.weight)
        if self.final_linear.bias is not None:
            nn.init.zeros_(self.final_linear.bias)

    def forward(self, x, use_layernorm=True):
        x_norm = self.layernorm(x) if (self.layernorm is not None and use_layernorm) else x
        b = self.base_activation(x_norm)
        s = self.feature_transform(x_norm)
        combined = self.base_scale * b + self.spline_scale * s
        return self.final_linear(combined)

class KAF(nn.Module):
    def __init__(self, layers_hidden, num_grids: int = 16, spline_dropout: float = 0.0, use_layernorm: bool = True):
        super().__init__()
        layers = []
        for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:]):
            layers.append(FastKAFLayer(in_dim, out_dim, num_grids=num_grids, spline_dropout=spline_dropout, use_layernorm=use_layernorm))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
