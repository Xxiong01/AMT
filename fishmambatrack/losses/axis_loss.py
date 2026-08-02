"""
fishmambatrack.losses.axis_loss

Axis alignment loss for π-periodic axis representation a=(cos2θ, sin2θ).

Given:
  pred_axis: (B,2)
  tgt_axis : (B,2)   (velocity-derived pseudo label)
  weight   : (B,)    reliability r in [0,1]

Loss:
  L = 1 - <pred_norm, tgt_norm>

Weighted mean:
  sum(r * L) / (sum(r) + eps)
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def axis_alignment_loss(
    pred_axis: torch.Tensor,
    tgt_axis: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    return_stats: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Args:
      pred_axis: (B,2) (not necessarily normalized)
      tgt_axis : (B,2) (usually already unit)
      weight   : (B,) reliability in [0,1]
    Returns:
      loss, stats
    """
    if pred_axis.ndim != 2 or pred_axis.size(1) != 2:
        raise ValueError(f"pred_axis must be (B,2), got {tuple(pred_axis.shape)}")
    if tgt_axis.ndim != 2 or tgt_axis.size(1) != 2:
        raise ValueError(f"tgt_axis must be (B,2), got {tuple(tgt_axis.shape)}")

    p = F.normalize(pred_axis, dim=1, eps=eps)
    t = F.normalize(tgt_axis, dim=1, eps=eps)

    dot = (p * t).sum(dim=1).clamp(-1.0, 1.0)  # (B,)
    per_loss = 1.0 - dot                       # (B,)

    if weight is None:
        loss = per_loss.mean()
        wsum = float(per_loss.numel())
        wmean = 1.0
        dot_mean = float(dot.mean().item())
        # angular error in degrees: Δ = 0.5 * arccos(dot)  (because dot=cos(2Δ))
        ang = 0.5 * torch.acos(dot.clamp(-1 + 1e-6, 1 - 1e-6)) * (180.0 / math.pi)
        ang_mean = float(ang.mean().item())
    else:
        w = weight.to(dtype=per_loss.dtype, device=per_loss.device)
        wsum_t = w.sum()
        # avoid NaN when a batch has all zeros
        loss = (per_loss * w).sum() / (wsum_t + eps)
        wsum = float(wsum_t.item())
        wmean = float(w.mean().item()) if w.numel() > 0 else 0.0

        dot_mean = float(((dot * w).sum() / (wsum_t + eps)).item())
        ang = 0.5 * torch.acos(dot.clamp(-1 + 1e-6, 1 - 1e-6)) * (180.0 / math.pi)
        ang_mean = float(((ang * w).sum() / (wsum_t + eps)).item())

    stats: Dict[str, float] = {}
    if return_stats:
        stats = {
            "axis_loss": float(loss.item()),
            "axis_dot_mean": float(dot_mean),
            "axis_ang_deg_mean": float(ang_mean),
            "axis_weight_sum": float(wsum),
            "axis_weight_mean": float(wmean),
        }
    return loss, stats
