import numpy as np
import torch
import torch.nn.functional as F

def psnr(gt, pred):
    # gt/pred: uint8 HxWxC or float 0..1
    if gt.dtype != np.float32 and gt.dtype != np.float64:
        gt = gt.astype(np.float32) / 255.0
    if pred.dtype != np.float32 and pred.dtype != np.float64:
        pred = pred.astype(np.float32) / 255.0
    mse = np.mean((gt - pred)**2)
    if mse == 0: return 99.0
    return 10.0 * np.log10(1.0 / mse)

def _gaussian(window_size, sigma):
    gauss = torch.tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)], dtype=torch.float32)
    return gauss/gauss.sum()

def _create_window(window_size, channel):
    _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window @ _1D_window.t()
    window = _2D_window.unsqueeze(0).unsqueeze(0)
    window = window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11):
    # img1/img2: np array HxWxC uint8 or float in 0..1
    if img1.dtype != np.float32 and img1.dtype != np.float64:
        img1 = img1.astype(np.float32) / 255.0
    if img2.dtype != np.float32 and img2.dtype != np.float64:
        img2 = img2.astype(np.float32) / 255.0
    x = torch.from_numpy(img1).permute(2,0,1).unsqueeze(0)
    y = torch.from_numpy(img2).permute(2,0,1).unsqueeze(0)
    channel = x.size(1)
    window = _create_window(window_size, channel)
    window = window.to(dtype=x.dtype)

    C1 = 0.01**2
    C2 = 0.03**2

    mu1 = F.conv2d(x, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(y, window, padding=window_size//2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2
    sigma1_sq = F.conv2d(x*x, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(y*y, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(x*y, window, padding=window_size//2, groups=channel) - mu1_mu2
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean().item())
