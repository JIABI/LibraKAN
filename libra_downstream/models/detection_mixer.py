import torch
import torch.nn as nn
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn

from mixers.mlp import MixerFFN as MLPFFN
from mixers.kaf import MixerFFN as KAFFFN
from mixers.librakan import MixerFFN as LibraFFN

MIXER_REG = {
    "mlp": MLPFFN,
    "kaf": KAFFFN,
    "librakan": LibraFFN,
}

class TwoMixerHead(nn.Module):
    def __init__(self, in_channels: int, representation_size: int, mixer_type: str = "mlp"):
        super().__init__()
        hidden = representation_size
        Mixer = MIXER_REG[mixer_type]
        self.flatten = nn.Flatten(start_dim=1)
        self.ffn = Mixer(in_dim=in_channels, hidden=hidden, out_dim=representation_size)

    def forward(self, x):
        x = self.flatten(x)
        x = self.ffn(x)
        return x

def build_model(task: str = "det", mixer: str = "mlp", pretrained: bool = True, **mixer_kwargs):
    assert task in {"det","seg"}
    if task == "det":
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT" if pretrained else None)
    else:
        model = maskrcnn_resnet50_fpn(weights="DEFAULT" if pretrained else None)

    rep_size = model.roi_heads.box_head.fc6.out_features if hasattr(model.roi_heads.box_head, "fc6") else 1024
    in_feats = model.roi_heads.box_head.fc6.in_features if hasattr(model.roi_heads.box_head, "fc6") else model.roi_heads.box_head.fc7.in_features
    model.roi_heads.box_head = TwoMixerHead(in_channels=in_feats, representation_size=rep_size, mixer_type=mixer)

    if task == "seg" and hasattr(model.roi_heads, "mask_head"):
        class MaskMixer(nn.Module):
            def __init__(self, base, mixer_type: str):
                super().__init__()
                self.base = base
                C = 256
                Mixer = MIXER_REG[mixer_type]
                self.mix = Mixer(in_dim=C, hidden=C, out_dim=C)
            def forward(self, x):
                x = self.base.conv5_mask(x)
                N, C, H, W = x.shape
                y = x.permute(0,2,3,1).reshape(-1, C)
                y = self.mix(y).view(N, H, W, C).permute(0,3,1,2).contiguous()
                x = torch.relu(y)
                return x
        mh = model.roi_heads.mask_head
        mh.forward = MaskMixer(mh, mixer).forward

    return model
