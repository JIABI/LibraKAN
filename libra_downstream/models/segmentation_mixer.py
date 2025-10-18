import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.ops import FeaturePyramidNetwork

from mixers.mlp import MixerFFN as MLPFFN
from mixers.kaf import MixerFFN as KAFFFN
from mixers.librakan import MixerFFN as LibraFFN

MIXER_REG = {"mlp": MLPFFN, "kaf": KAFFFN, "librakan": LibraFFN}

class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        m = resnet50(weights="DEFAULT" if pretrained else None)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4
        self.out_channels = [256, 512, 1024, 2048]

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x) 
        c3 = self.layer2(c2) 
        c4 = self.layer3(c3) 
        c5 = self.layer4(c4) 
        return {"0": c2, "1": c3, "2": c4, "3": c5}

class UPerLikeHead(nn.Module):
    def __init__(self, in_channels_list, out_channels=256, mixer_type="mlp", num_classes=150):
        super().__init__()
        self.fpn = FeaturePyramidNetwork(in_channels_list, out_channels)
        Mixer = MIXER_REG[mixer_type]
        self.mlp = Mixer(in_dim=out_channels, hidden=out_channels, out_dim=out_channels)
        self.cls = nn.Conv2d(out_channels, num_classes, kernel_size=1)

    def forward(self, feats):
        x = self.fpn(feats)
        p2 = x["0"]
        N, C, H, W = p2.shape
        y = p2.permute(0,2,3,1).reshape(-1, C)
        y = self.mlp(y).view(N, H, W, C).permute(0,3,1,2).contiguous()
        logits = self.cls(y)
        return logits

class SegModel(nn.Module):
    def __init__(self, mixer_type="mlp", pretrained_backbone=True, num_classes=150):
        super().__init__()
        self.backbone = ResNetBackbone(pretrained=pretrained_backbone)
        self.decode_head = UPerLikeHead(self.backbone.out_channels, out_channels=256, mixer_type=mixer_type, num_classes=num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.decode_head(feats)
        return logits
