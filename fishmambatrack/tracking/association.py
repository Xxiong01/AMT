"""
fishmambatrack.tracking.association

Association utilities:
- IoU for tlwh
- Appearance distance: prefer EMA feature (trk.feature) for stability
- Motion cost: Mahalanobis^2 on center (cx,cy) ONLY
  * Can be used as soft cost even when maha_gate is disabled
  * Separate maha_gate (hard gating) and maha_cost_scale (soft cost scaling)

This design is important for fish:
- Fish bbox w/h changes a lot during rotation, so we never use (w,h) for motion gating.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .tracklet import Tracklet, tlwh_to_xywh
from .kalman_filter import KalmanFilter


def iou_tlwh(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
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


def appearance_distance(track: Tracklet, det_feat: np.ndarray) -> Tuple[float, bool]:
    """
    Return:
      dist = 1 - cosine_sim  (smaller is better)
      has_feat
    IMPORTANT: prefer EMA feature for stability, fall back to bank only if EMA missing.
    """
    if det_feat is None:
        return 0.0, False

    if track.feature is not None:
        return float(1.0 - float(np.dot(track.feature, det_feat))), True

    if track.feature_bank and len(track.feature_bank) > 0:
        best_dot = -1.0
        for f in track.feature_bank:
            d = float(np.dot(f, det_feat))
            if d > best_dot:
                best_dot = d
        return float(1.0 - best_dot), True

    return 0.0, False


def _maha_xy(kf: KalmanFilter, trk: Tracklet, det_xy: np.ndarray) -> np.ndarray:
    """
    Mahalanobis^2 distance on center only: (cx,cy)
    det_xy: (N,2)
    """
    mean_z, cov_z = kf.project(trk.mean, trk.cov)  # (4,), (4,4)
    mean_xy = mean_z[:2].astype(np.float32)
    cov_xy = cov_z[:2, :2].astype(np.float32) + np.eye(2, dtype=np.float32) * 1e-6

    chol = np.linalg.cholesky(cov_xy).astype(np.float32)
    d = (det_xy - mean_xy[None, :]).astype(np.float32)     # (N,2)
    z = np.linalg.solve(chol, d.T).astype(np.float32)      # (2,N)
    return np.sum(z * z, axis=0)                           # (N,)


def build_cost_matrix_v3(
    kf: KalmanFilter,
    tracks: Sequence[Tracklet],
    det_tlwh: np.ndarray,            # (N,4)
    det_feat: Optional[np.ndarray],  # (N,D)
    *,
    w_app: float = 0.7,
    w_iou: float = 0.3,
    w_motion: float = 0.2,
    iou_gate: float = 0.2,
    maha_gate: float = 0.0,          # <=0 disables HARD gating
    maha_cost_scale: float = 25.0,   # used for SOFT motion cost normalization
    app_gate: Optional[float] = None,
    large_cost: float = 1e6,
) -> np.ndarray:
    """
    cost = w_app * app_dist + w_iou * (1 - iou) + w_motion * motion_cost
    motion_cost = clamp(maha_xy / maha_cost_scale, 0..1)

    Gates:
      - IoU gate always applied
      - if maha_gate>0: maha_xy <= maha_gate (hard gate)
      - if app_gate is not None and track has feat: app_dist <= app_gate
    """
    T = len(tracks)
    N = int(det_tlwh.shape[0])
    cost = np.full((T, N), fill_value=large_cost, dtype=np.float32)
    if T == 0 or N == 0:
        return cost

    det_xywh = np.stack(
        [tlwh_to_xywh(tuple(map(float, det_tlwh[i].tolist()))) for i in range(N)],
        axis=0
    ).astype(np.float32)
    det_xy = det_xywh[:, :2]

    for i, trk in enumerate(tracks):
        maha = None
        need_motion = (w_motion is not None and float(w_motion) > 0.0) or (maha_gate is not None and float(maha_gate) > 0.0)
        if need_motion:
            maha = _maha_xy(kf, trk, det_xy)  # (N,)

        for j in range(N):
            # hard motion gate (optional)
            if maha is not None and (maha_gate is not None) and float(maha_gate) > 0.0:
                if float(maha[j]) > float(maha_gate):
                    continue

            t_box = trk.to_tlwh()
            d_box = tuple(map(float, det_tlwh[j].tolist()))
            iou = iou_tlwh(t_box, d_box)
            if iou < float(iou_gate):
                continue
            iou_dist = 1.0 - iou

            # appearance
            if det_feat is None:
                app_dist = 0.0
                has_feat = False
            else:
                app_dist, has_feat = appearance_distance(trk, det_feat[j])

            if has_feat and (app_gate is not None):
                if float(app_dist) > float(app_gate):
                    continue

            # soft motion cost
            motion_cost = 0.0
            if maha is not None and (maha_cost_scale is not None) and float(maha_cost_scale) > 1e-9:
                motion_cost = float(min(1.0, float(maha[j]) / float(maha_cost_scale)))

            ww_app = float(w_app) if has_feat else 0.0
            ww_iou = float(w_iou)
            ww_m = float(w_motion) if (maha is not None) else 0.0

            cost[i, j] = ww_app * float(app_dist) + ww_iou * float(iou_dist) + ww_m * float(motion_cost)

    return cost


def hungarian_with_threshold(cost: np.ndarray, thresh: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    T, N = cost.shape
    if T == 0:
        return [], [], list(range(N))
    if N == 0:
        return [], list(range(T)), []

    row_ind, col_ind = linear_sum_assignment(cost)
    matches: List[Tuple[int, int]] = []
    unmatched_t = set(range(T))
    unmatched_d = set(range(N))

    for r, c in zip(row_ind.tolist(), col_ind.tolist()):
        if cost[r, c] <= float(thresh):
            matches.append((r, c))
            unmatched_t.discard(r)
            unmatched_d.discard(c)

    return matches, sorted(list(unmatched_t)), sorted(list(unmatched_d))
