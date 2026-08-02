"""
fishmambatrack.models.reid.temporal_baseline_reid

Controlled temporal ReID baselines with a shared frame encoder (ResNet-18):
  - mean_pool
  - gru
  - lstm
  - transformer_lite
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_resnet18(pretrained: bool = False, weights_path: Optional[str] = None) -> nn.Module:
    try:
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if (pretrained and not weights_path) else None
        m = resnet18(weights=weights)
    except Exception:
        from torchvision.models import resnet18

        m = resnet18(pretrained=bool(pretrained and not weights_path))

    if weights_path:
        path = Path(weights_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Backbone weights not found: {path}")
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        if isinstance(state, dict):
            m.load_state_dict(state, strict=False)

    m.fc = nn.Identity()
    return m


@dataclass
class FishTemporalBaselineReIDConfig:
    pretrained_backbone: bool = False
    backbone_weights_path: Optional[str] = None

    temporal_type: str = "mean_pool"  # mean_pool | gru | lstm | transformer_lite
    d_model: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    nhead: int = 4
    ff_mult: int = 2

    use_pos_embed: bool = True
    max_seq_len: int = 32
    pool_mode: str = "mean_last"  # mean | last | mean_last

    infer_repeat_len: int = 1

    emb_dim: int = 256
    use_bnneck: bool = True
    num_classes: int = 0


class FishTemporalBaselineReID(nn.Module):
    def __init__(self, cfg: Optional[FishTemporalBaselineReIDConfig] = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = FishTemporalBaselineReIDConfig()
        self.cfg = cfg

        temporal_type = str(cfg.temporal_type).lower()
        if temporal_type not in {"mean_pool", "gru", "lstm", "transformer_lite"}:
            raise ValueError(f"Unsupported temporal_type='{cfg.temporal_type}'.")
        self.temporal_type = temporal_type

        self.backbone = _build_resnet18(
            pretrained=cfg.pretrained_backbone,
            weights_path=cfg.backbone_weights_path,
        )

        d_model = int(cfg.d_model)
        self.proj_in = nn.Linear(512, d_model)

        if self.temporal_type == "gru":
            self.temporal = nn.GRU(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=int(cfg.num_layers),
                dropout=float(cfg.dropout) if int(cfg.num_layers) > 1 else 0.0,
                batch_first=True,
            )
        elif self.temporal_type == "lstm":
            self.temporal = nn.LSTM(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=int(cfg.num_layers),
                dropout=float(cfg.dropout) if int(cfg.num_layers) > 1 else 0.0,
                batch_first=True,
            )
        elif self.temporal_type == "transformer_lite":
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=int(cfg.nhead),
                dim_feedforward=int(d_model * int(cfg.ff_mult)),
                dropout=float(cfg.dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(layer, num_layers=int(cfg.num_layers))
        else:
            self.temporal = nn.Identity()

        self.use_pos_embed = bool(cfg.use_pos_embed)
        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, int(cfg.max_seq_len), d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

        self.proj_out = nn.Linear(d_model, int(cfg.emb_dim))
        self.use_bnneck = bool(cfg.use_bnneck)
        self.bnneck = nn.BatchNorm1d(int(cfg.emb_dim)) if self.use_bnneck else nn.Identity()

        self.num_classes = int(cfg.num_classes)
        self.classifier = nn.Linear(int(cfg.emb_dim), self.num_classes, bias=False) if self.num_classes > 0 else None

    def _add_pos_embed(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.pos_embed is None:
            return tokens
        _, t, _ = tokens.shape
        if t <= self.pos_embed.shape[1]:
            return tokens + self.pos_embed[:, :t, :]
        pe = self.pos_embed.transpose(1, 2)
        pe = F.interpolate(pe, size=t, mode="linear", align_corners=False).transpose(1, 2)
        return tokens + pe

    def _pool(self, tokens: torch.Tensor) -> torch.Tensor:
        mode = str(self.cfg.pool_mode).lower()
        if mode == "last":
            return tokens[:, -1]
        if mode == "mean":
            return tokens.mean(dim=1)
        if mode == "mean_last":
            return 0.5 * (tokens.mean(dim=1) + tokens[:, -1])
        raise ValueError(f"Unknown pool_mode='{self.cfg.pool_mode}'.")

    def forward(self, x: torch.Tensor, *, return_logits: bool = False) -> Dict[str, Any]:
        if x.ndim == 4:
            rep = max(1, int(getattr(self.cfg, "infer_repeat_len", 1)))
            x = x.unsqueeze(1)
            if rep > 1:
                x = x.repeat(1, rep, 1, 1, 1)
        if x.ndim != 5:
            raise ValueError(f"Expected x (B,T,3,H,W), got {tuple(x.shape)}")

        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        feat = self.backbone(x).reshape(b, t, -1)  # (B,T,512)

        tokens = self.proj_in(feat)
        if self.temporal_type in {"gru", "lstm", "transformer_lite"}:
            tokens = self._add_pos_embed(tokens)
            if self.temporal_type in {"gru", "lstm"}:
                tokens, _ = self.temporal(tokens)
            else:
                tokens = self.temporal(tokens)

        pooled = tokens.mean(dim=1) if self.temporal_type == "mean_pool" else self._pool(tokens)

        emb = self.proj_out(pooled)
        emb_bn = self.bnneck(emb)
        emb_norm = F.normalize(emb, dim=1, eps=1e-6)

        out: Dict[str, Any] = {
            "emb": emb_norm,
            "emb_bn": emb_bn,
        }
        if return_logits and self.classifier is not None:
            out["logits"] = self.classifier(emb_bn)
        return out
