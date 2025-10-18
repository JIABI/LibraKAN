import torch
from thop import profile

def count_params_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

@torch.no_grad()
def get_flops_g(model, example_input):
    macs, params = profile(model, inputs=(example_input,), verbose=False)
    return float(macs / 1e9)
