"""
fishmambatrack.models.reid.common

Shared helpers for ReID models that support axis-guided alignment.

This module is intentionally free of Mamba dependencies so that baseline
models can be imported even when `mamba_ssm` is not installed.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_resnet(name: str = "resnet18", *, pretrained: bool = False) -> nn.Module:
    name = str(name).lower()
    if name not in ("resnet18", "resnet50"):
        raise ValueError(f"Unsupported backbone '{name}'. Use 'resnet18' or 'resnet50'.")

    try:
        if name == "resnet18":
            from torchvision.models import resnet18, ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            m = resnet18(weights=weights)
        else:
            from torchvision.models import resnet50, ResNet50_Weights
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            m = resnet50(weights=weights)
    except Exception:
        if name == "resnet18":
            from torchvision.models import resnet18
            m = resnet18(pretrained=pretrained)
        else:
            from torchvision.models import resnet50
            m = resnet50(pretrained=pretrained)

    m.fc = nn.Identity()
    return m


def axis_to_theta(axis: torch.Tensor) -> torch.Tensor:
    """
    axis: (B,2) = (cos2θ, sin2θ) (not necessarily normalized)
    Return θ in radians, shape (B,).
    Note: θ and θ+π are equivalent (axis periodicity).
    """
    a = F.normalize(axis, dim=1, eps=1e-6)
    theta2 = torch.atan2(a[:, 1], a[:, 0])
    return 0.5 * theta2


def resolve_axis_theta_used(
    *,
    axis_pred: torch.Tensor,
    theta_pred: torch.Tensor,
    axis_override: Optional[torch.Tensor] = None,
    theta_override: Optional[torch.Tensor] = None,
    override_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Returns: axis_used, theta_used, mask (bool tensor or None)
    """
    if axis_pred.ndim != 2 or axis_pred.shape[1] != 2:
        raise ValueError(f"axis_pred must be (B,2), got {tuple(axis_pred.shape)}")
    if theta_pred.ndim != 1:
        raise ValueError(f"theta_pred must be (B,), got {tuple(theta_pred.shape)}")

    B = int(axis_pred.shape[0])
    device = axis_pred.device

    theta_used = theta_pred
    axis_used = axis_pred

    if override_mask is not None:
        mask = override_mask.to(device=device).bool()
        if mask.ndim == 0:
            mask = mask.expand(B)
        if mask.numel() != B:
            raise ValueError(f"override_mask must be (B,), got {tuple(mask.shape)} B={B}")
    else:
        mask = None

    if theta_override is not None:
        th = theta_override.to(device=device)
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
            ax = F.normalize(axis_override.to(device=device), dim=1, eps=1e-6)
            if mask is None:
                axis_used = ax
            else:
                axis_used = axis_pred.clone()
                axis_used[mask] = ax[mask]

    elif axis_override is not None:
        ax = F.normalize(axis_override.to(device=device), dim=1, eps=1e-6)
        th = axis_to_theta(ax)
        if mask is None:
            axis_used = ax
            theta_used = th
        else:
            axis_used = axis_pred.clone()
            theta_used = theta_pred.clone()
            axis_used[mask] = ax[mask]
            theta_used[mask] = th[mask]

    return axis_used, theta_used, mask


def apply_reverse_mask(
    tokens: torch.Tensor,
    *,
    reverse_mask: Optional[torch.Tensor] = None,
    override_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Reverse token order (dim=1) for selected samples.
    tokens: (B,L,C)
    reverse_mask: (B,) bool
    override_mask: (B,) bool, if provided we only reverse where override_mask is True.
    """
    if reverse_mask is None:
        return tokens

    rm = reverse_mask.to(device=tokens.device).bool()
    if rm.ndim == 0:
        rm = rm.expand(tokens.shape[0])
    if override_mask is not None:
        rm = rm & override_mask.to(device=tokens.device).bool()

    if bool(rm.any().item()):
        tokens = tokens.clone()
        tokens[rm] = torch.flip(tokens[rm], dims=[1])
    return tokens


def cfg_to_dict(cfg: Any) -> Dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    if is_dataclass(cfg):
        return asdict(cfg)
    raise TypeError(f"Unsupported cfg type: {type(cfg)}")

