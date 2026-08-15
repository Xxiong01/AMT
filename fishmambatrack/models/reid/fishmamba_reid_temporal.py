"""ResNet-18 plus Mamba temporal ReID encoder used by AMT-L48."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fishmambatrack.models.backbones.mamba_blocks import (
    MambaEncoder,
    MambaEncoderConfig,
)
from fishmambatrack.models.reid.common import forward_backbone


def _build_resnet18(pretrained: bool) -> nn.Module:
    try:
        from torchvision.models import ResNet18_Weights, resnet18

        model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    except Exception:
        from torchvision.models import resnet18

        model = resnet18(pretrained=pretrained)
    model.fc = nn.Identity()
    return model


@dataclass
class FishMambaReIDTemporalConfig:
    mamba_d_model: int = 256
    mamba_layers: int = 2
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_dropout: float = 0.10
    max_seq_len: int = 48
    emb_dim: int = 256
    num_classes: int = 0


class FishMambaReIDTemporal(nn.Module):
    def __init__(self, cfg: Optional[FishMambaReIDTemporalConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or FishMambaReIDTemporalConfig()
        self.backbone = _build_resnet18(pretrained=self.cfg.num_classes > 0)
        self.proj_in = nn.Linear(512, self.cfg.mamba_d_model)
        self.mamba = MambaEncoder(
            MambaEncoderConfig(
                d_model=self.cfg.mamba_d_model,
                n_layers=self.cfg.mamba_layers,
                d_state=self.cfg.mamba_d_state,
                d_conv=self.cfg.mamba_d_conv,
                expand=self.cfg.mamba_expand,
                dropout=self.cfg.mamba_dropout,
            )
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.cfg.max_seq_len, self.cfg.mamba_d_model)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.proj_out = nn.Linear(self.cfg.mamba_d_model, self.cfg.emb_dim)
        self.bnneck = nn.BatchNorm1d(self.cfg.emb_dim)
        self.classifier = (
            nn.Linear(self.cfg.emb_dim, self.cfg.num_classes, bias=False)
            if self.cfg.num_classes > 0
            else None
        )

    def _add_pos_embed(self, tokens: torch.Tensor) -> torch.Tensor:
        length = tokens.shape[1]
        if length <= self.pos_embed.shape[1]:
            return tokens + self.pos_embed[:, :length]
        values = F.interpolate(
            self.pos_embed.transpose(1, 2),
            size=length,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        return tokens + values

    @staticmethod
    def _pool(tokens: torch.Tensor) -> torch.Tensor:
        return 0.5 * (tokens.mean(dim=1) + tokens[:, -1])

    def encode_frame_features(self, frame_features: torch.Tensor) -> torch.Tensor:
        """Encode a sequence of precomputed ResNet-18 frame features."""
        tokens = self._add_pos_embed(self.proj_in(frame_features))
        embedding = self.proj_out(self._pool(self.mamba(tokens)))
        return F.normalize(embedding, dim=1, eps=1e-6)

    def forward(
        self, images: torch.Tensor, *, return_logits: bool = False
    ) -> Dict[str, Any]:
        if images.ndim == 4:
            images = images.unsqueeze(1)
        if images.ndim != 5:
            raise ValueError(f"Expected (B,T,3,H,W), got {tuple(images.shape)}")
        batch, length, channels, height, width = images.shape
        frame_features = forward_backbone(
            self,
            self.backbone,
            images.reshape(batch * length, channels, height, width),
        ).reshape(batch, length, -1)
        tokens = self._add_pos_embed(self.proj_in(frame_features))
        embedding = self.proj_out(self._pool(self.mamba(tokens)))
        embedding_bn = self.bnneck(embedding)
        output: Dict[str, Any] = {
            "emb": F.normalize(embedding, dim=1, eps=1e-6),
            "emb_bn": embedding_bn,
        }
        if return_logits:
            if self.classifier is None:
                raise RuntimeError(
                    "The classification head is available only during training."
                )
            output["logits"] = self.classifier(embedding_bn)
        return output
