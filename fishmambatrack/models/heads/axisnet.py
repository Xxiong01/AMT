"""
fishmambatrack.models.heads.axisnet

AxisNet: predict fish body-axis representation a=(cos(2θ), sin(2θ))
- π-periodic (θ and θ+π are equivalent)
- Designed to be trained with velocity-derived pseudo labels
- Can operate on images directly (default) with a small backbone
- Uses torchvision ResNet18 if available, otherwise a tiny CNN fallback

Output:
  axis: (B,2) normalized to unit length
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _try_build_resnet18(pretrained: bool = False) -> Tuple[Optional[nn.Module], Optional[int]]:
    """
    Try to build torchvision resnet18 as a feature extractor.
    Return (model, feat_dim) or (None, None) if torchvision not available.
    """
    try:
        import torchvision  # noqa: F401
        try:
            # torchvision>=0.13/0.14/0.15 style
            from torchvision.models import resnet18, ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            m = resnet18(weights=weights)
        except Exception:
            # older style
            from torchvision.models import resnet18
            m = resnet18(pretrained=pretrained)

        # Remove classifier
        m.fc = nn.Identity()
        return m, 512
    except Exception:
        return None, None


class TinyCNN(nn.Module):
    """
    Very small CNN fallback (no torchvision dependency).
    Input: (B,3,H,W)
    Output: (B,feat_dim)
    """
    def __init__(self, feat_dim: int = 256) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(256, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        x = self.proj(x)
        return x


@dataclass
class AxisNetConfig:
    backbone: str = "resnet18"  # resnet18 | tinycnn
    pretrained: bool = False
    feat_dim: int = 512         # used for tinycnn proj dim, ignored by resnet18
    hidden_dim: int = 256
    dropout: float = 0.0


class AxisNet(nn.Module):
    """
    Predict axis representation a=(cos2θ, sin2θ).

    Forward input:
      - image tensor (B,3,H,W)  -> uses internal backbone
      - feature tensor (B,C) or (B,C,H,W) -> if you want to reuse another backbone later

    By default we use internal backbone on images.
    """
    def __init__(self, cfg: Optional[AxisNetConfig] = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = AxisNetConfig()

        self.cfg = cfg
        self.backbone_name = cfg.backbone.lower()

        if self.backbone_name == "resnet18":
            resnet, feat_dim = _try_build_resnet18(pretrained=cfg.pretrained)
            if resnet is None:
                # fallback
                self.backbone = TinyCNN(feat_dim=cfg.feat_dim)
                self.feat_dim = cfg.feat_dim
                self.backbone_name = "tinycnn"
            else:
                self.backbone = resnet
                self.feat_dim = int(feat_dim)
        elif self.backbone_name == "tinycnn":
            self.backbone = TinyCNN(feat_dim=cfg.feat_dim)
            self.feat_dim = cfg.feat_dim
        else:
            raise ValueError(f"Unknown backbone '{cfg.backbone}'. Use 'resnet18' or 'tinycnn'.")

        hd = int(cfg.hidden_dim)
        dp = float(cfg.dropout)

        self.head = nn.Sequential(
            nn.Linear(self.feat_dim, hd),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dp) if dp > 0 else nn.Identity(),
            nn.Linear(hd, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
          axis_pred: (B,2) unit-normalized
        """
        if x.dim() == 4:
            # Image input -> use backbone to get (B,feat_dim)
            feat = self.backbone(x)
            if feat.dim() == 4:
                feat = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
        elif x.dim() == 2:
            # Feature vector input
            feat = x
        else:
            raise ValueError(f"AxisNet input must be (B,3,H,W) or (B,C). Got shape={tuple(x.shape)}")

        axis = self.head(feat)                 # (B,2)
        axis = F.normalize(axis, dim=1, eps=1e-6)
        return axis
