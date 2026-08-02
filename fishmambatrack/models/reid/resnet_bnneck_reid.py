"""
fishmambatrack.models.reid.resnet_bnneck_reid

Classic ReID baseline:
  ResNet (18/50) + projection + BNNeck + classifier.

Axis head is provided for compatibility with the rest of the project, but
the embedding is not axis-aligned (pure global pooling baseline).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fishmambatrack.models.reid.common import axis_to_theta, build_resnet


@dataclass
class ResNetBNNeckReIDConfig:
    backbone: str = "resnet50"  # resnet18 | resnet50
    pretrained_backbone: bool = False

    emb_dim: int = 256
    use_bnneck: bool = True

    axis_hidden: int = 256
    axis_dropout: float = 0.0

    num_classes: int = 0


class ResNetBNNeckReID(nn.Module):
    def __init__(self, cfg: Optional[ResNetBNNeckReIDConfig] = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = ResNetBNNeckReIDConfig()
        self.cfg = cfg

        self.backbone = build_resnet(cfg.backbone, pretrained=cfg.pretrained_backbone)
        self.backbone_name = str(cfg.backbone).lower()

        feat_dim = 2048 if self.backbone_name == "resnet50" else 512

        hd = int(cfg.axis_hidden)
        dp = float(cfg.axis_dropout)
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hd),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dp) if dp > 0 else nn.Identity(),
            nn.Linear(hd, 2),
        )

        self.proj = nn.Linear(feat_dim, int(cfg.emb_dim))

        self.use_bnneck = bool(cfg.use_bnneck)
        if self.use_bnneck:
            self.bnneck = nn.BatchNorm1d(int(cfg.emb_dim))
        else:
            self.bnneck = nn.Identity()

        self.num_classes = int(cfg.num_classes)
        if self.num_classes > 0:
            self.classifier = nn.Linear(int(cfg.emb_dim), self.num_classes, bias=False)
        else:
            self.classifier = None

    @torch.no_grad()
    def load_axisnet_checkpoint(self, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        incompatible = self.load_state_dict(state, strict=False)
        print(f"[ResNetBNNeckReID] Loaded AxisNet ckpt: {ckpt_path}")
        print(f"  Missing keys: {len(incompatible.missing_keys)}")
        print(f"  Unexpected keys: {len(incompatible.unexpected_keys)}")

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_logits: bool = False,
        return_debug: bool = False,
        axis_override: Optional[torch.Tensor] = None,
        theta_override: Optional[torch.Tensor] = None,
        override_mask: Optional[torch.Tensor] = None,
        reverse_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        feat = self.backbone(x)
        if feat.dim() == 4:
            feat = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)

        axis_raw = self.head(feat)
        axis_pred = F.normalize(axis_raw, dim=1, eps=1e-6)
        theta_pred = axis_to_theta(axis_pred)

        axis_used = axis_pred
        theta_used = theta_pred
        if theta_override is not None:
            th = theta_override.to(device=x.device)
            if th.ndim == 0:
                th = th.expand(x.shape[0])
            if override_mask is None:
                theta_used = th
            else:
                mk = override_mask.to(device=x.device).bool()
                if mk.ndim == 0:
                    mk = mk.expand(x.shape[0])
                theta_used = theta_pred.clone()
                theta_used[mk] = th[mk]

        if axis_override is not None:
            ax = F.normalize(axis_override.to(device=x.device), dim=1, eps=1e-6)
            if override_mask is None:
                axis_used = ax
                theta_used = axis_to_theta(ax) if theta_override is None else theta_used
            else:
                mk = override_mask.to(device=x.device).bool()
                if mk.ndim == 0:
                    mk = mk.expand(x.shape[0])
                axis_used = axis_pred.clone()
                axis_used[mk] = ax[mk]
                if theta_override is None:
                    th = axis_to_theta(ax)
                    theta_used = theta_pred.clone()
                    theta_used[mk] = th[mk]

        emb_raw = self.proj(feat)
        emb_bn = self.bnneck(emb_raw)
        emb = F.normalize(emb_bn, dim=1, eps=1e-6)

        out: Dict[str, Any] = {
            "emb": emb,
            "emb_bn": emb_bn,
            "axis": axis_pred,
            "theta": theta_pred,
            "axis_used": axis_used,
            "theta_used": theta_used,
        }

        if return_logits and (self.classifier is not None):
            out["logits"] = self.classifier(emb_bn)

        if return_debug:
            out["feat_shape"] = tuple(feat.shape)

        return out

