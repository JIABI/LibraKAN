import torch

def p_from_plogit(plogit):
    return torch.sigmoid(plogit)

def generalized_shrinkage(z_raw, lambda_param, p):
    if not torch.is_tensor(p):
        p = torch.tensor(p, device=z_raw.device, dtype=z_raw.dtype)
    lam = lambda_param if isinstance(lambda_param, torch.Tensor) else torch.tensor(lambda_param, device=z_raw.device, dtype=z_raw.dtype)
    r = z_raw.abs()
    z = torch.sign(z_raw) * torch.clamp(r - lam * (r ** (1.0 - p)), min=0.0)
    return z
