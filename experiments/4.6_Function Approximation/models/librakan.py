
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Shrinkage(nn.Module):
    def __init__(self, init_lambda=0.05, init_p=0.6):
        super().__init__()
        self.lam_raw = nn.Parameter(torch.tensor(init_lambda).log())
        self.p_raw   = nn.Parameter(torch.logit(torch.tensor(init_p)))

    def forward(self, z):
        lam = F.softplus(self.lam_raw)
        p   = torch.sigmoid(self.p_raw)
        return torch.sign(z) * torch.relu(torch.abs(z) - lam).pow(p)

class LibraKAN1D(nn.Module):
    def __init__(self, spectral_dim=64, act="gelu", k_active=16, fmax=8.0, es_beta_init=7.0):
        super().__init__()
        self.spectral_dim = spectral_dim
        self.k_active = k_active
        self.fmax = fmax
        Act = nn.GELU if act.lower()=="gelu" else nn.ReLU
        self.local = nn.Sequential(nn.Linear(1,64), Act(), nn.Linear(64,32), Act())
        init = torch.linspace(0.5, 8.0, spectral_dim).log()
        self.freqs_raw = nn.Parameter(init)
        self.phase     = nn.Parameter(torch.zeros(spectral_dim))
        self.es_beta_raw = nn.Parameter(torch.tensor(es_beta_init).log())
        self.gate = nn.Parameter(torch.zeros(spectral_dim))
        self.shrink = Shrinkage(init_lambda=0.0005, init_p=0.6)
        self.mix_in  = nn.Linear(1, spectral_dim, bias=False)
        self.mix_out = nn.Linear(spectral_dim, 32, bias=False)
        nn.init.xavier_uniform_(self.mix_in.weight)
        nn.init.xavier_uniform_(self.mix_out.weight)
        self.head = nn.Linear(64, 1)  # 32 local + 32 spectral
        self._decor = torch.tensor(0.0)
        self._gl = torch.tensor(0.0)

    def es_beta(self):
        return F.softplus(self.es_beta_raw).clamp(2.0, 20.0)

    def es_window(self, f_abs):
        r = (f_abs / (self.fmax + 1e-8)).clamp(0, 1)
        inside = (1 - r*r).clamp(0, 1)
        return torch.exp(self.es_beta() * (torch.sqrt(inside + 1e-8) - 1.0))

    def spectral_features(self, x):
        u = 5*self.mix_in(x)                                   # (B,F)
        f = F.softplus(self.freqs_raw)                       # (F,)
        z = 2*math.pi * (u * f.unsqueeze(0) + self.phase.unsqueeze(0))
        s = torch.cos(z)
        s = self.shrink(s)
        g = torch.sigmoid(self.gate)
        s = s * g.unsqueeze(0)
        if self.k_active is not None and self.k_active < self.spectral_dim:
            with torch.no_grad():
                score = s.abs().mean(0)
                _, idx = torch.topk(score, self.k_active)
                mask = torch.zeros_like(score); mask[idx] = 1.0
            s = s * mask.unsqueeze(0)
        w_es = self.es_window(f)
        s = s * w_es.unsqueeze(0)
        S = s - s.mean(0, keepdim=True)
        C = (S.T @ S) / (S.shape[0] + 1e-6)
        self._decor = ((C - C.diag().diag())**2).mean()
        self._gl = g.abs().sum()
        h = self.mix_out(s)                                  # (B,32)
        return h

    def forward(self, x):
        h_loc = self.local(x)
        h_spec = self.spectral_features(x)
        return self.head(torch.cat([h_loc, h_spec], dim=-1))

    def extra_losses(self):
        return {"group_lasso": self._gl, "decor": self._decor}

class MLP(nn.Module):
    def __init__(self, act="relu", width=64, depth=2):
        super().__init__()
        Act = nn.ReLU if act=="relu" else nn.GELU
        layers = []; in_dim = 1
        for _ in range(depth):
            layers += [nn.Linear(in_dim, width), Act()]
            in_dim = width
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)
