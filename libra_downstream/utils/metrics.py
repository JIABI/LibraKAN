import torch
import numpy as np

def compute_miou(pred, target, num_classes: int):
    pred = pred.detach().cpu().numpy().astype(np.int64)
    gt = target.detach().cpu().numpy().astype(np.int64)
    ious = []
    for cls in range(num_classes):
        p = (pred == cls)
        g = (gt == cls)
        inter = (p & g).sum()
        union = (p | g).sum()
        if union > 0:
            ious.append(inter / union)
    if len(ious) == 0:
        return 0.0
    return float(np.mean(ious) * 100.0)
