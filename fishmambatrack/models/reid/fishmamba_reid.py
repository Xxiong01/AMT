"""
fishmambatrack.models.reid.fishmamba_reid

FishMambaReID (v2): support velocity-teacher forcing for alignment/scan.

Key additions:
- forward supports:
    axis_override: (B,2) optional, used to compute theta for alignment
    theta_override: (B,) optional, directly used for alignment
    override_mask: (B,) bool optional, apply override only on masked samples
- Outputs keep backward compatible keys:
    "axis"  : predicted axis (cos2θ,sin2θ), normalized
    "theta" : predicted θ (radians)
  plus:
    "axis_used"  : axis actually used for alignment (pred or override)
    "theta_used" : theta actually used for alignment
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fishmambatrack.models.align.roi_rotate import rotate_tensor
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
        if isinstance(state, dict):
            if any(k.startswith("module.") for k in state.keys()):
                state = {k.replace("module.", "", 1): v for k, v in state.items()}
        m.load_state_dict(state, strict=False)

    m.fc = nn.Identity()
    return m


def axis_to_theta(axis: torch.Tensor) -> torch.Tensor:
    """
    axis: (B,2) = (cos2θ, sin2θ) (not necessarily normalized)
    Return θ in radians, shape (B,).
    Note: θ and θ+π are equivalent (axis periodicity).
    """
    a = F.normalize(axis, dim=1, eps=1e-6)
    theta2 = torch.atan2(a[:, 1], a[:, 0])   # 2θ
    theta = 0.5 * theta2
    return theta


@dataclass
class FishMambaReIDConfig:
    pretrained_backbone: bool = False
    backbone_weights_path: Optional[str] = None

    # tokenization
    num_tokens: int = 16

    # mamba
    mamba_d_model: int = 256
    mamba_layers: int = 2
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_dropout: float = 0.0

    # positional encoding (helps sequence models)
    use_pos_embed: bool = False

    # pooling over sequence output
    # - "mean": mean pool (baseline)
    # - "last": last token (direction-aware when reverse_mask is used)
    # - "mean_last": 0.5 * (mean + last)
    pool_mode: str = "mean"

    # embedding
    emb_dim: int = 256
    use_bnneck: bool = True

    # axis head (match AxisNet style: 512->hidden->2)
    axis_hidden: int = 256
    axis_dropout: float = 0.0

    # training classifier
    num_classes: int = 0


class FishMambaReID(nn.Module):
    def __init__(self, cfg: Optional[FishMambaReIDConfig] = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = FishMambaReIDConfig()
        self.cfg = cfg

        self.backbone = _build_resnet18(
            pretrained=cfg.pretrained_backbone,
            weights_path=cfg.backbone_weights_path,
        )

        # Axis head name保持为 head（兼容 AxisNet ckpt）
        hd = int(cfg.axis_hidden)
        dp = float(cfg.axis_dropout)
        self.head = nn.Sequential(
            nn.Linear(512, hd),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dp) if dp > 0 else nn.Identity(),
            nn.Linear(hd, 2),
        )

        self.proj_in = nn.Linear(256, int(cfg.mamba_d_model))
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
            self.pos_embed = nn.Parameter(torch.zeros(1, int(cfg.num_tokens), int(cfg.mamba_d_model)))
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

    @torch.no_grad()
    def load_axisnet_checkpoint(self, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        incompatible = self.load_state_dict(state, strict=False)
        print(f"[FishMambaReID] Loaded AxisNet ckpt: {ckpt_path}")
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
        feat_map = b.layer3(x)              # (B,256,16,8) for 256x128 input
        x = b.layer4(feat_map)              # (B,512,8,4)

        feat_global = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return feat_map, feat_global

    def _tokenize(self, feat_map: torch.Tensor) -> torch.Tensor:
        """
        feat_map: (B,256,H,W)
        tokens:   (B,L,256), mean over height, L along width
        """
        B, C, H, W = feat_map.shape
        L = int(self.cfg.num_tokens)
        if W != L:
            feat_map = F.interpolate(feat_map, size=(H, L), mode="bilinear", align_corners=False)
        tokens = feat_map.mean(dim=2).transpose(1, 2).contiguous()
        return tokens

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_logits: bool = False,
        return_debug: bool = False,
        axis_override: Optional[torch.Tensor] = None,     # (B,2)
        theta_override: Optional[torch.Tensor] = None,    # (B,) or scalar
        override_mask: Optional[torch.Tensor] = None,     # (B,) bool
        reverse_mask: Optional[torch.Tensor] = None,      # (B,) bool
    ) -> Dict[str, Any]:
        """
        Input:
          x: (B,3,H,W)

        Teacher forcing / velocity-guided alignment:
          - Provide axis_override (cos2θ,sin2θ) from velocity pseudo label
          - Provide override_mask to apply only on confident samples
          - Model still predicts axis (out["axis"]) for supervision

        Output:
          emb: L2-normalized embedding for matching
          axis/theta: predicted
          axis_used/theta_used: used for rotation alignment (pred or override)
        """
        feat_map, feat_global = self._forward_backbone(x)

        axis_raw = self.head(feat_global)
        axis_pred = F.normalize(axis_raw, dim=1, eps=1e-6)  # (B,2)
        theta_pred = axis_to_theta(axis_pred)               # (B,)

        # Decide which theta/axis to use for alignment
        B = x.shape[0]
        theta_used = theta_pred
        axis_used = axis_pred

        if override_mask is not None:
            mask = override_mask.to(device=x.device).bool()
            if mask.numel() != B:
                raise ValueError(f"override_mask must be (B,), got {tuple(mask.shape)} B={B}")
        else:
            mask = None

        if theta_override is not None:
            th = theta_override.to(device=x.device)
            if th.ndim == 0:
                th = th.expand(B)
            elif th.ndim == 1 and th.numel() == 1:
                th = th.expand(B)
            elif th.ndim != 1 or th.numel() != B:
                raise ValueError(f"theta_override must be scalar or (B,), got {tuple(th.shape)} B={B}")

            if mask is None:
                theta_used = th
            else:
                theta_used = theta_pred.clone()
                theta_used[mask] = th[mask]

            if axis_override is not None:
                ax = F.normalize(axis_override.to(device=x.device), dim=1, eps=1e-6)
                if mask is None:
                    axis_used = ax
                else:
                    axis_used = axis_pred.clone()
                    axis_used[mask] = ax[mask]

        elif axis_override is not None:
            ax = F.normalize(axis_override.to(device=x.device), dim=1, eps=1e-6)
            th = axis_to_theta(ax)

            if mask is None:
                axis_used = ax
                theta_used = th
            else:
                axis_used = axis_pred.clone()
                theta_used = theta_pred.clone()
                axis_used[mask] = ax[mask]
                theta_used[mask] = th[mask]

        # Align fish axis to horizontal: rotate by -theta_used
        feat_rot = rotate_tensor(feat_map, angle_rad=-theta_used)

        tokens = self._tokenize(feat_rot)           # (B,L,256)
        # Direction-aware scan: reverse token order for selected samples
        if reverse_mask is not None:
            rm = reverse_mask.to(device=x.device).bool()
            if rm.ndim == 0:
                rm = rm.expand(x.shape[0])
            # safer: only reverse where override_mask is True (if provided)
            if override_mask is not None:
                rm = rm & override_mask.to(device=x.device).bool()

            if bool(rm.any().item()):
                tokens = tokens.clone()
                tokens[rm] = torch.flip(tokens[rm], dims=[1])
        tokens = self.proj_in(tokens)               # (B,L,D)
        if self.pos_embed is not None:
            pos = self.pos_embed
            if int(pos.shape[1]) != int(tokens.shape[1]):
                # Safety: adapt to different token length (shouldn't happen if num_tokens matches input width).
                # pos: (1,L,D) -> (1,D,L) for interpolation -> (1,L',D)
                pos = F.interpolate(
                    pos.transpose(1, 2),
                    size=int(tokens.shape[1]),
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
            tokens = tokens + pos.to(device=tokens.device, dtype=tokens.dtype)
        tokens = self.mamba(tokens)                 # (B,L,D)

        pool_mode = str(getattr(self.cfg, "pool_mode", "mean")).lower()
        if pool_mode == "mean":
            pooled = tokens.mean(dim=1)
        elif pool_mode == "last":
            pooled = tokens[:, -1]
        elif pool_mode == "mean_last":
            pooled = 0.5 * tokens.mean(dim=1) + 0.5 * tokens[:, -1]
        else:
            raise ValueError(f"Unknown pool_mode='{self.cfg.pool_mode}' (expected mean/last/mean_last)")

        emb_raw = self.proj_out(pooled)             # (B,emb_dim)

        emb_bn = self.bnneck(emb_raw)
        emb = F.normalize(emb_bn, dim=1, eps=1e-6)

        out: Dict[str, Any] = {
            "emb": emb,
            "emb_bn": emb_bn,
            "axis": axis_pred,          # backward compatible: predicted axis
            "theta": theta_pred,        # predicted theta
            "axis_used": axis_used,     # used for alignment
            "theta_used": theta_used,   # used for alignment
        }

        if return_logits and (self.classifier is not None):
            out["logits"] = self.classifier(emb_bn)

        if return_debug:
            out["feat_map_shape"] = tuple(feat_map.shape)
            out["tokens_shape"] = tuple(tokens.shape)
            if mask is not None:
                out["override_frac"] = float(mask.float().mean().item())

        return out
