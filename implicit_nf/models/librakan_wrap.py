# models/librakan_wrap.py

import torch

import torch.nn as nn

from .librakan_mixer import LibraKANMixer 


class LibraKANImplicit(nn.Module):
    def __init__(self,

                 in_dim: int = 2,

                 hidden: int = 128,

                 depth: int = 6,

                 out_dim: int = 3,

                 **libra_kwargs):

        super().__init__()

        self.inp = nn.Linear(in_dim, hidden)

        blocks = []

        mixers = []

        for _ in range(depth - 1):
            blocks.append(nn.Linear(hidden, hidden))

            m = LibraKANMixer(width=hidden, **libra_kwargs) 

            mixers.append(m)

            blocks.extend([m, nn.GELU()])

        self.blocks = nn.Sequential(*blocks)

        self.mixers = nn.ModuleList(mixers)

        self.head = nn.Linear(hidden, out_dim)

        self._tau_active = 1e-3

    # -------- forward --------

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = self.inp(coords)

        x = self.blocks(x)

        x = self.head(x)

        return x

    # -------- losses / stats --------

    def extra_losses(self):
        device = next(self.parameters()).device

        l1 = torch.zeros((), device=device)

        bal = torch.zeros((), device=device)

        for m in self.mixers:

            if hasattr(m, "aux_losses"):
                aux = m.aux_losses()

                l1 = l1 + aux.get("l1_spec", 0.0)

                bal = bal + aux.get("balance", 0.0)

        return {"l1_spectral": l1, "balance": bal}

    @torch.no_grad()
    def active_freq(self, tau: float = None) -> int:
        thr = float(self._tau_active if tau is None else tau)

        cnt = 0

        for m in self.mixers:

            if hasattr(m, "effective_alpha"):
                eff = m.effective_alpha().abs()  # (F, W)

                cnt += int((eff > thr).sum().item())

        return int(cnt)


    def spectral_parameters(self):

         params = []

        for m in self.mixers:

            if hasattr(m, "spectral_params"):
                params.extend(list(m.spectral_params()))

        return params

    def p_parameters(self):
        params = []

        for m in self.mixers:

            if hasattr(m, "p_logit"):
                params.append(m.p_logit)

        return params

    def local_parameters(self):
        spec_ids = set(id(p) for p in self.spectral_parameters())

        p_ids = set(id(p) for p in self.p_parameters())

        params = []

        # inp + blocks(linear) + head

        for mod in [self.inp, *self.blocks, self.head]:

            for p in mod.parameters(recurse=False):

                if id(p) not in spec_ids and id(p) not in p_ids:
                    params.append(p)

        return params

    def param_groups(self, base_lr: float, spec_lr: float, p_lr: float):
        groups = [

            {"params": self.local_parameters(), "lr": base_lr, "name": "local"},

            {"params": self.spectral_parameters(), "lr": spec_lr, "name": "spectral"},

        ]

        p_params = self.p_parameters()

        if len(p_params) > 0:
            groups.append({"params": p_params, "lr": p_lr, "name": "p"})

        return groups


