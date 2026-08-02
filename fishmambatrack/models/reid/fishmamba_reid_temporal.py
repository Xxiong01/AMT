"""
fishmambatrack.models.reid.fishmamba_reid_temporal

Tracklet-level temporal ReID: CNN per frame + Mamba over time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fishmambatrack.models.backbones.mamba_blocks import MambaEncoder, MambaEncoderConfig


def _build_resnet18(pretrained: bool = False, weights_path: Optional[str] = None) -> nn.Module:
    try:
        from torchvision.models import resnet18, ResNet18_Weights
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
class FishMambaReIDTemporalConfig:
    pretrained_backbone: bool = False
    backbone_weights_path: Optional[str] = None

    # mamba
    mamba_d_model: int = 256
    mamba_layers: int = 2
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_dropout: float = 0.0

    # positional encoding
    use_pos_embed: bool = True
    max_seq_len: int = 32

    # pooling
    pool_mode: str = "mean_last"  # mean | last | mean_last

    # inference fallback for frame-only callers:
    # if input is (B,3,H,W), repeat to (B,infer_repeat_len,3,H,W)
    infer_repeat_len: int = 1

    # embedding
    emb_dim: int = 256
    use_bnneck: bool = True

    # classifier
    num_classes: int = 0


class FishMambaReIDTemporal(nn.Module):
    def __init__(self, cfg: Optional[FishMambaReIDTemporalConfig] = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = FishMambaReIDTemporalConfig()
        self.cfg = cfg

        self.backbone = _build_resnet18(
            pretrained=cfg.pretrained_backbone,
            weights_path=cfg.backbone_weights_path,
        )

        self.proj_in = nn.Linear(512, int(cfg.mamba_d_model))
        self.mamba = MambaEncoder(
            MambaEncoderConfig(
                d_model=int(cfg.mamba_d_model),
                n_layers=int(cfg.mamba_layers),
                d_state=int(cfg.mamba_d_state),
                d_conv=int(cfg.mamba_d_conv),
                expand=int(cfg.mamba_expand),
                dropout=float(cfg.mamba_dropout),
            )
        )
        self.proj_out = nn.Linear(int(cfg.mamba_d_model), int(cfg.emb_dim))

        self.use_pos_embed = bool(cfg.use_pos_embed)
        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, int(cfg.max_seq_len), int(cfg.mamba_d_model)))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

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

    def _add_pos_embed(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.pos_embed is None:
            return tokens
        b, t, d = tokens.shape
        if t <= self.pos_embed.shape[1]:
            return tokens + self.pos_embed[:, :t, :]
        pe = self.pos_embed.transpose(1, 2)
        pe = F.interpolate(pe, size=t, mode="linear", align_corners=False).transpose(1, 2)
        return tokens + pe

    def _pool(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.cfg.pool_mode == "last":
            return tokens[:, -1]
        if self.cfg.pool_mode == "mean":
            return tokens.mean(dim=1)
        if self.cfg.pool_mode == "mean_last":
            return 0.5 * (tokens.mean(dim=1) + tokens[:, -1])
        raise ValueError(f"Unknown pool_mode='{self.cfg.pool_mode}'")

    def forward(self, x: torch.Tensor, *, return_logits: bool = False) -> Dict[str, Any]:
        """
        x: (B,T,3,H,W) or (B,3,H,W)
        """
        if x.ndim == 4:
            rep = max(1, int(getattr(self.cfg, "infer_repeat_len", 1)))
            x = x.unsqueeze(1)
            if rep > 1:
                x = x.repeat(1, rep, 1, 1, 1)
        if x.ndim != 5:
            raise ValueError(f"Expected x (B,T,3,H,W), got {tuple(x.shape)}")

        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        feat = self.backbone(x)  # (B*T,512)
        feat = feat.reshape(b, t, -1)

        tokens = self.proj_in(feat)
        tokens = self._add_pos_embed(tokens)
        tokens = self.mamba(tokens)
        pooled = self._pool(tokens)

        emb = self.proj_out(pooled)
        emb_bn = self.bnneck(emb) if self.use_bnneck else emb
        emb_norm = F.normalize(emb, dim=1, eps=1e-6)

        out: Dict[str, Any] = {
            "emb": emb_norm,
            "emb_bn": emb_bn,
        }
        if return_logits and self.classifier is not None:
            out["logits"] = self.classifier(emb_bn)
        return out
