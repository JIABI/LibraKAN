import torch

def _es_window(t, beta=6.0, T=3.0, eps=1e-6):
    mask = (t.abs() <= T).to(t.dtype)
    val = torch.clamp(1 - (t / T) ** 2, min=0.0) + eps
    return torch.exp(beta * (torch.sqrt(val) - 1.0)) * mask

def _grid_gather_indices(pos, K, M):
    left = torch.floor(pos - K/2)
    idxs = [(left + k).long() % M for k in range(K)]
    dists = [pos - idxs[k] for k in range(K)]
    return idxs, dists

def _nufft1d(u, epsilon, beta=6.0, fmax=None, oversamp=2, grid_T=3.0):
    N, F = u.shape
    device, dtype = u.device, u.dtype
    if fmax is not None:
        u = torch.clamp(u, min=-abs(fmax), max=abs(fmax))
    M = oversamp * 512
    pos = (u % 1.0) * M
    K = 8
    g = torch.zeros(N, M, device=device, dtype=dtype)
    idxs, dists = _grid_gather_indices(pos, K, M)
    for k in range(K):
        t = dists[k] / (M / 2) * grid_T
        w = _es_window(t, beta=beta, T=grid_T) * epsilon.unsqueeze(0)
        g.scatter_add_(1, idxs[k], w)
    g_fft = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(g, dim=1), dim=1), dim=1).real
    z = torch.zeros(N, F, device=device, dtype=dtype)
    for k in range(K):
        t = dists[k] / (M / 2) * grid_T
        w = _es_window(t, beta=beta, T=grid_T)
        z += w * g_fft.gather(1, idxs[k])
    return z

def _nufft2d(u, epsilon, beta=6.0, fmax=None, oversamp=2, grid_T=3.0):
    N, F, two = u.shape
    assert two == 2, "u must be (N,F,2)"
    device, dtype = u.device, u.dtype
    if fmax is not None:
        u = torch.clamp(u, min=-abs(fmax), max=abs(fmax))
    ux = u[...,0] % 1.0
    uy = u[...,1] % 1.0
    M = oversamp * 256
    K = 6
    posx = ux * M
    posy = uy * M
    idxsx, distsx = _grid_gather_indices(posx, K, M)
    idxsy, distsy = _grid_gather_indices(posy, K, M)
    G = torch.zeros(N, M, M, device=device, dtype=dtype)
    for kx in range(K):
        tx = distsx[kx] / (M / 2) * grid_T
        wx = _es_window(tx, beta=beta, T=grid_T)  # (N,F)
        Ix = idxsx[kx]
        for ky in range(K):
            ty = distsy[ky] / (M / 2) * grid_T
            wy = _es_window(ty, beta=beta, T=grid_T)  # (N,F)
            Iy = idxsy[ky]
            w = (wx * wy) * epsilon.unsqueeze(0)  # (N,F)
            # Accumulate into rows then columns (approx separable scatter)
            rowbuf = torch.zeros(N, M, device=device, dtype=dtype)
            rowbuf.scatter_add_(1, Ix, w)
            G.scatter_add_(2, Iy.unsqueeze(1).expand(-1,M,-1), rowbuf.unsqueeze(1).expand(-1,-1,Iy.size(1)))
    Gfft = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(G, dim=(1,2)), dim=(1,2)), dim=(1,2)).real
    z = torch.zeros(N, F, device=device, dtype=dtype)
    for kx in range(K):
        tx = distsx[kx] / (M / 2) * grid_T
        wx = _es_window(tx, beta=beta, T=grid_T); Ix = idxsx[kx]
        rows = Gfft.gather(1, Ix.unsqueeze(-1).expand(-1,-1,M))
        for ky in range(K):
            ty = distsy[ky] / (M / 2) * grid_T
            wy = _es_window(ty, beta=beta, T=grid_T); Iy = idxsy[ky]
            cols = rows.gather(2, Iy.unsqueeze(-1)).squeeze(-1)
            z += (wx * wy) * cols
    return z

def nufft_es_forward(u, epsilon, beta=6.0, fmax=None, oversamp=2, grid_T=3.0):
    if u.dim() == 2:
        return _nufft1d(u, epsilon, beta=beta, fmax=fmax, oversamp=oversamp, grid_T=grid_T)
    elif u.dim() == 3 and u.size(-1) == 2:
        return _nufft2d(u, epsilon, beta=beta, fmax=fmax, oversamp=oversamp, grid_T=grid_T)
    else:
        raise ValueError("u must be (N,F) for 1D or (N,F,2) for 2D")
