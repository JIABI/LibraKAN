
import torch
import numpy as np
from PIL import Image, ImageDraw
import torch.nn as nn
import torch.optim as optim
from typing import Tuple

def set_seed_all(seed:int):
    import random, os
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def make_coord_grid(H:int, W:int, device=None):
    q = torch.linspace(-1.0, 1.0, H, device=device)
    e = torch.linspace(-1.0, 1.0, W, device=device)
    qq, ee = torch.meshgrid(q, e, indexing="ij")
    coords = torch.stack([ee, qq], dim=-1).view(-1,2)  # (H*W,2) -> (x,y)
    return coords

def image_to_tensor01(img_np: np.ndarray) -> torch.Tensor:
    if img_np.ndim == 2:
        img_np = img_np[..., None]
    img = torch.from_numpy(img_np).float() / 255.0
    img = img.permute(2,0,1).unsqueeze(0).contiguous()  # (1,C,H,W)
    return img

def tensor01_to_image(img: torch.Tensor) -> np.ndarray:
    img = img.clamp(0,1).detach().cpu()[0].permute(1,2,0).numpy()
    return (img*255.0 + 0.5).astype(np.uint8)

def save_image01(img: torch.Tensor, path: str):
    from PIL import Image
    im = Image.fromarray(tensor01_to_image(img))
    im.save(path)

def cosine_lr(optimizer, lr_max: float, lr_min: float, T: int):
    import numpy as np
    from torch.optim.lr_scheduler import LambdaLR
    def lr_lambda(step):
        s = min(step, T)
        return lr_min/lr_max + 0.5*(1 - lr_min/lr_max)*(1 + np.cos(np.pi * s / T))
    return LambdaLR(optimizer, lr_lambda)

def crop_visuals(gt: torch.Tensor, pred: torch.Tensor, out_path: str, crops=((0.25,0.25),(0.5,0.5),(0.75,0.75)), size=96):
    gt_np = tensor01_to_image(gt)
    pr_np = tensor01_to_image(pred)
    H, W = gt_np.shape[:2]
    out_w = size*len(crops)*2
    out_h = size*1
    canvas = Image.new("RGB", (out_w, out_h), (255,255,255))
    xoff = 0
    for cy, cx in crops:
        cxp = int(cx*W); cyp = int(cy*H)
        x0 = max(0, cxp-size//2); x1 = min(W, x0+size)
        y0 = max(0, cyp-size//2); y1 = min(H, y0+size)
        gt_c = gt_np[y0:y1, x0:x1]
        pr_c = pr_np[y0:y1, x0:x1]
        gt_im = Image.fromarray(gt_c).resize((size,size), Image.BICUBIC)
        pr_im = Image.fromarray(pr_c).resize((size,size), Image.BICUBIC)
        canvas.paste(gt_im, (xoff, 0))
        canvas.paste(pr_im, (xoff+size, 0))
        xoff += 2*size
    canvas.save(out_path)
