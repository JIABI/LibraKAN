
import torch
import torch.nn.functional as F


def psnr_torch(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0, eps: float = 1e-12) -> torch.Tensor:
    """
    pred, target: (N,C,H,W) in [0,1]
      """
    if pred.dim() == 3:  # (C,H,W) -> (1,C,H,W)
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    mse = ((pred - target) ** 2).flatten(1).mean(dim=1)  # (N,)
    psnr = 10.0 * torch.log10((max_val ** 2) / (mse + eps))  # (N,)
    return psnr.mean()


def ssim_torch(img1: torch.Tensor, img2: torch.Tensor, C1: float = 0.01**2, C2: float = 0.03**2, window_size:int=11):
    import torch
    import torch.nn.functional as F
    from math import exp

    def gaussian(window_size, sigma):
        gauss = torch.tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = (_1D_window @ _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
