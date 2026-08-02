"""
fishmambatrack.models.align.roi_rotate

Rotation alignment using affine_grid + grid_sample.
This rotates a tensor around its center.

Convention:
  rotate_tensor(x, angle_rad) returns output rotated by +angle_rad (CCW),
  i.e. content is rotated counter-clockwise by angle_rad.

Implementation detail:
  affine_grid expects a matrix that maps output coords -> input coords,
  so we use -angle internally.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def rotate_tensor(
    x: torch.Tensor,
    angle_rad: torch.Tensor,
    *,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> torch.Tensor:
    """
    Args:
      x: (B,C,H,W)
      angle_rad: (B,) or scalar tensor, desired CCW rotation angle
    Returns:
      rotated x with same shape
    """
    if x.ndim != 4:
        raise ValueError(f"x must be (B,C,H,W), got {tuple(x.shape)}")

    B, C, H, W = x.shape
    if not torch.is_tensor(angle_rad):
        angle_rad = torch.tensor(angle_rad, device=x.device, dtype=x.dtype)

    if angle_rad.ndim == 0:
        angle_rad = angle_rad.expand(B)
    elif angle_rad.ndim == 1 and angle_rad.numel() == 1:
        angle_rad = angle_rad.expand(B)
    elif angle_rad.ndim != 1 or angle_rad.numel() != B:
        raise ValueError(f"angle_rad must be scalar or (B,), got shape={tuple(angle_rad.shape)} and B={B}")

    # We want output rotated by +angle (CCW).
    # grid_sample uses theta mapping output->input, so use -angle here.
    a = -angle_rad

    cos_a = torch.cos(a)
    sin_a = torch.sin(a)

    theta = torch.zeros((B, 2, 3), device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = cos_a
    theta[:, 0, 1] = -sin_a
    theta[:, 1, 0] = sin_a
    theta[:, 1, 1] = cos_a
    # theta[:, :, 2] = 0 (no translation)

    grid = F.affine_grid(theta, size=x.size(), align_corners=align_corners)
    y = F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    return y
