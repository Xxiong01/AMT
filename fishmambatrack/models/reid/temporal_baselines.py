"""Controlled temporal ReID baselines sharing the AMT ResNet-18 backbone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fishmambatrack.models.reid.common import forward_backbone


def _resnet18(pretrained: bool) -> nn.Module:
    try:
        from torchvision.models import ResNet18_Weights, resnet18

        model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    except Exception:
        from torchvision.models import resnet18

        model = resnet18(pretrained=pretrained)
    model.fc = nn.Identity()
    return model


@dataclass
class TemporalBaselineConfig:
    name: str = "mean_pool"
    temporal_d_model: int = 256
    temporal_layers: int = 2
    temporal_nhead: int = 4
    temporal_ff_mult: int = 2
    temporal_dropout: float = 0.10
    max_seq_len: int = 48
    emb_dim: int = 256
    num_classes: int = 0


class TemporalBaselineReID(nn.Module):
    """Mean, recurrent, Transformer, or single-frame temporal control."""

    def __init__(self, cfg: Optional[TemporalBaselineConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or TemporalBaselineConfig()
        self.backbone = _resnet18(pretrained=self.cfg.num_classes > 0)
        self.proj_in = nn.Linear(512, self.cfg.temporal_d_model)
        name = self.cfg.name
        if name == "gru":
            self.temporal = nn.GRU(
                self.cfg.temporal_d_model,
                self.cfg.temporal_d_model,
                num_layers=self.cfg.temporal_layers,
                dropout=(
                    self.cfg.temporal_dropout if self.cfg.temporal_layers > 1 else 0.0
                ),
                batch_first=True,
            )
        elif name == "lstm":
            self.temporal = nn.LSTM(
                self.cfg.temporal_d_model,
                self.cfg.temporal_d_model,
                num_layers=self.cfg.temporal_layers,
                dropout=(
                    self.cfg.temporal_dropout if self.cfg.temporal_layers > 1 else 0.0
                ),
                batch_first=True,
            )
        elif name == "transformer_lite":
            layer = nn.TransformerEncoderLayer(
                d_model=self.cfg.temporal_d_model,
                nhead=self.cfg.temporal_nhead,
                dim_feedforward=self.cfg.temporal_d_model * self.cfg.temporal_ff_mult,
                dropout=self.cfg.temporal_dropout,
                batch_first=True,
                norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(layer, self.cfg.temporal_layers)
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.cfg.max_seq_len, self.cfg.temporal_d_model)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        elif name in {"mean_pool", "single_frame"}:
            self.temporal = nn.Identity()
        else:
            raise ValueError(f"Unsupported temporal baseline: {name}")
        self.proj_out = nn.Linear(self.cfg.temporal_d_model, self.cfg.emb_dim)
        self.bnneck = nn.BatchNorm1d(self.cfg.emb_dim)
        self.classifier = (
            nn.Linear(self.cfg.emb_dim, self.cfg.num_classes, bias=False)
            if self.cfg.num_classes > 0
            else None
        )

    def _aggregate(self, tokens: torch.Tensor) -> torch.Tensor:
        name = self.cfg.name
        if name == "single_frame":
            return tokens[:, -1]
        if name == "mean_pool":
            return tokens.mean(dim=1)
        if name in {"gru", "lstm"}:
            output, _ = self.temporal(tokens)
            return output[:, -1]
        length = tokens.shape[1]
        if length <= self.pos_embed.shape[1]:
            position = self.pos_embed[:, :length]
        else:
            position = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=length,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        return self.temporal(tokens + position).mean(dim=1)

    def encode_frame_features(self, frame_features: torch.Tensor) -> torch.Tensor:
        """Encode a sequence of precomputed ResNet-18 frame features."""
        embedding = self.proj_out(self._aggregate(self.proj_in(frame_features)))
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
        embedding = self.proj_out(self._aggregate(self.proj_in(frame_features)))
        output: Dict[str, Any] = {
            "emb": F.normalize(embedding, dim=1, eps=1e-6),
            "emb_bn": self.bnneck(embedding),
        }
        if return_logits:
            if self.classifier is None:
                raise RuntimeError(
                    "The classification head is available only during training."
                )
            output["logits"] = self.classifier(output["emb_bn"])
        return output
