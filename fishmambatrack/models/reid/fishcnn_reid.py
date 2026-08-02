"""
fishmambatrack.models.reid.fishcnn_reid

Baseline ReID model:
  ResNet18 + axis head + rotation alignment + tokenization + mean pooling.

This isolates the benefit of the sequence model (Mamba/Transformer) by
removing it while keeping the same input alignment and tokenization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fishmambatrack.models.align.roi_rotate import rotate_tensor
from fishmambatrack.models.reid.common import (
    apply_reverse_mask,
    axis_to_theta,
    build_resnet,
    resolve_axis_theta_used,
)


@dataclass
class FishCNNReIDConfig:
    pretrained_backbone: bool = False
    num_tokens: int = 16
    emb_dim: int = 256
    use_bnneck: bool = True

    axis_hidden: int = 256
    axis_dropout: float = 0.0

    num_classes: int = 0


class FishCNNReID(nn.Module):
    def __init__(self, cfg: Optional[FishCNNReIDConfig] = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = FishCNNReIDConfig()
        self.cfg = cfg

        self.backbone = build_resnet("resnet18", pretrained=cfg.pretrained_backbone)

        hd = int(cfg.axis_hidden)
        dp = float(cfg.axis_dropout)
        self.head = nn.Sequential(
            nn.Linear(512, hd),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dp) if dp > 0 else nn.Identity(),
            nn.Linear(hd, 2),
        )

        self.proj_out = nn.Linear(256, int(cfg.emb_dim))

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
        print(f"[FishCNNReID] Loaded AxisNet ckpt: {ckpt_path}")
        print(f"  Missing keys: {len(incompatible.missing_keys)}")
        print(f"  Unexpected keys: {len(incompatible.unexpected_keys)}")

    def _forward_backbone(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          feat_map: layer3 output (B,256,H,W) for tokenization
          feat_global: layer4 pooled (B,512) for axis head
        """
        b = self.backbone
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.relu(x)
        x = b.maxpool(x)

        x = b.layer1(x)
        x = b.layer2(x)
        feat_map = b.layer3(x)
        x = b.layer4(feat_map)

        feat_global = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return feat_map, feat_global

    def _tokenize(self, feat_map: torch.Tensor) -> torch.Tensor:
        """
        feat_map: (B,256,H,W)
        tokens: (B,L,256), mean over height, L along width
        """
        B, C, H, W = feat_map.shape
        L = int(self.cfg.num_tokens)
        if W != L:
            feat_map = F.interpolate(feat_map, size=(H, L), mode="bilinear", align_corners=False)
        return feat_map.mean(dim=2).transpose(1, 2).contiguous()

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
        feat_map, feat_global = self._forward_backbone(x)

        axis_raw = self.head(feat_global)
        axis_pred = F.normalize(axis_raw, dim=1, eps=1e-6)
        theta_pred = axis_to_theta(axis_pred)

        axis_used, theta_used, mask = resolve_axis_theta_used(
            axis_pred=axis_pred,
            theta_pred=theta_pred,
            axis_override=axis_override,
            theta_override=theta_override,
            override_mask=override_mask,
        )

        feat_rot = rotate_tensor(feat_map, angle_rad=-theta_used)
        tokens = self._tokenize(feat_rot)
        tokens = apply_reverse_mask(tokens, reverse_mask=reverse_mask, override_mask=override_mask)

        pooled = tokens.mean(dim=1)
        emb_raw = self.proj_out(pooled)

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
            out["feat_map_shape"] = tuple(feat_map.shape)
            out["tokens_shape"] = tuple(tokens.shape)
            if mask is not None:
                out["override_frac"] = float(mask.float().mean().item())

        return out

