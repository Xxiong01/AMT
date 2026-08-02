"""
fishmambatrack.tracking.velocity_guidance

Estimate per-detection velocity direction from previous frame detections,
and convert it into axis representation (cos2θ, sin2θ) for π-periodic orientation.

This is used for velocity-guided alignment/scan at inference time.

Design:
- For each current detection, find the highest-overlap previous detection by IoU.
- Reject if highest IoU too small or ambiguous (best - second_highest < margin).
- Compute displacement of centers -> angle phi.
- axis = (cos(2phi), sin(2phi))
- Reliability weight uses normalized speed: ||d|| / sqrt(area),
  mapped to [0,1] by (r_s0, r_s1). Only apply override if weight >= w_min.
"""

from __future__ import annotations
from typing import Optional, Tuple

import numpy as np


def _iou_tlwh(a: np.ndarray, b: np.ndarray) -> float:
    ax, ay, aw, ah = map(float, a.tolist())
    bx, by, bw, bh = map(float, b.tolist())
    ax1, ay1, ax2, ay2 = ax, ay, ax + aw, ay + ah
    bx1, by1, bx2, by2 = bx, by, bx + bw, by + bh

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def _center_xy(tlwh: np.ndarray) -> Tuple[float, float]:
    x, y, w, h = map(float, tlwh.tolist())
    return (x + 0.5 * w, y + 0.5 * h)


def velocity_axis_override(
    prev_tlwh: Optional[np.ndarray],   # (Nprev,4)
    curr_tlwh: np.ndarray,             # (N,4)
    *,
    iou_match: float = 0.20,
    iou_margin: float = 0.05,
    r_s0: float = 0.01,
    r_s1: float = 0.05,
    w_min: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      axis_override: (N,2) float32
      mask: (N,) bool
      weight: (N,) float32   (0..1, confidence of velocity direction)
    """
    N = int(curr_tlwh.shape[0])
    axis = np.zeros((N, 2), dtype=np.float32)
    mask = np.zeros((N,), dtype=np.bool_)
    wts = np.zeros((N,), dtype=np.float32)

    if prev_tlwh is None or int(prev_tlwh.shape[0]) == 0 or N == 0:
        return axis, mask, wts

    P = int(prev_tlwh.shape[0])

    for j in range(N):
        c = curr_tlwh[j]
        # compute IoUs with all prev boxes
        ious = np.zeros((P,), dtype=np.float32)
        for i in range(P):
            ious[i] = _iou_tlwh(c, prev_tlwh[i])

        top_idx = int(np.argmax(ious))
        top_iou = float(ious[top_idx])
        if top_iou < float(iou_match):
            continue

        if P >= 2:
            # second best iou
            sec = float(np.partition(ious, -2)[-2])
            if (top_iou - sec) < float(iou_margin):
                continue

        # displacement
        cx, cy = _center_xy(c)
        px, py = _center_xy(prev_tlwh[top_idx])
        dx = cx - px
        dy = cy - py
        speed = float(np.hypot(dx, dy))

        # normalized by sqrt(area)
        w, h = float(c[2]), float(c[3])
        area = max(w * h, 1.0)
        speed_norm = speed / (float(np.sqrt(area)) + 1e-6)

        # map to [0,1]
        if r_s1 > r_s0:
            ww = (speed_norm - float(r_s0)) / (float(r_s1) - float(r_s0))
        else:
            ww = 1.0 if speed_norm >= float(r_s0) else 0.0
        ww = float(np.clip(ww, 0.0, 1.0))
        wts[j] = ww

        if ww < float(w_min):
            continue

        phi = float(np.arctan2(dy, dx))
        axis[j, 0] = float(np.cos(2.0 * phi))
        axis[j, 1] = float(np.sin(2.0 * phi))
        mask[j] = True

    return axis, mask, wts
