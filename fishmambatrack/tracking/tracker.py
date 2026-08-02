"""
fishmambatrack.tracking.tracker

Tracker v3.1 (anti-drift):
- Matching stays the same (Hungarian on cost matrix)
- BUT feature update is frozen on ambiguous frames to avoid identity drift.
Ambiguity is detected by:
  (A) detection-detection overlap (collision IoU)
  (B) assignment margin in the cost matrix (best vs 2nd best too close)

This is particularly helpful for fish tracking with frequent crossings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Set

import numpy as np

from .kalman_filter import KalmanFilter
from .tracklet import Tracklet, tlwh_to_xywh
from .association import build_cost_matrix_v3, hungarian_with_threshold, appearance_distance, iou_tlwh


@dataclass
class FishTrackerConfig:
    max_age: int = 30
    min_hits: int = 1
    ema_alpha: float = 0.9

    # weights
    w_app: float = 0.7
    w_iou: float = 0.3
    w_motion: float = 0.0

    # gates / thresholds
    iou_gate: float = 0.2
    maha_gate: float = 0.0
    maha_cost_scale: float = 25.0
    app_gate: Optional[float] = None
    match_thresh: float = 0.75

    # detection filter
    min_det_score: float = 0.0

    # feature update control (prevents drift)
    feature_update_gate: float = 1.0     # if app_dist > gate -> do not update feature
    feature_bank_size: int = 1

    # ---- NEW: ambiguity freeze ----
    freeze_on_collision: bool = True
    collision_iou: float = 0.15          # det-det IoU > this -> ambiguous det -> freeze feature update

    freeze_on_margin: bool = True
    cost_margin: float = 0.05            # if (2nd_best - assigned) < margin -> ambiguous -> freeze update
    valid_cost_thresh: float = 1e5       # large cost means invalid


class FishTracker:
    def __init__(self, cfg: Optional[FishTrackerConfig] = None) -> None:
        self.cfg = cfg if cfg is not None else FishTrackerConfig()
        self.kf = KalmanFilter()
        self.tracks: List[Tracklet] = []
        self.next_id = 1
        self.frame_id = 0

    def reset(self) -> None:
        self.tracks = []
        self.next_id = 1
        self.frame_id = 0

    def _create_track(self, tlwh: Tuple[float, float, float, float], score: float, feat: Optional[np.ndarray]) -> None:
        z = tlwh_to_xywh(tlwh)
        mean, cov = self.kf.initiate(z)
        confirmed = (self.cfg.min_hits <= 1)

        trk = Tracklet(
            track_id=self.next_id,
            mean=mean,
            cov=cov,
            feature=None,
            score=float(score),
            confirmed=confirmed,
        )
        trk.update(
            self.kf, tlwh, score, feat,
            ema_alpha=self.cfg.ema_alpha,
            min_hits=self.cfg.min_hits,
            update_feature=True,
            bank_size=self.cfg.feature_bank_size,
        )
        self.tracks.append(trk)
        self.next_id += 1

    def _ambiguous_dets_by_collision(self, det_tlwh: np.ndarray) -> Set[int]:
        amb: Set[int] = set()
        if not self.cfg.freeze_on_collision:
            return amb
        thr = float(self.cfg.collision_iou)
        if thr <= 0.0:
            return amb
        N = int(det_tlwh.shape[0])
        for i in range(N):
            bi = tuple(map(float, det_tlwh[i].tolist()))
            for j in range(i + 1, N):
                bj = tuple(map(float, det_tlwh[j].tolist()))
                if iou_tlwh(bi, bj) > thr:
                    amb.add(i)
                    amb.add(j)
        return amb

    def _row_margin(self, cost: np.ndarray, ti: int, di: int) -> float:
        row = cost[ti]
        valid = row < float(self.cfg.valid_cost_thresh)
        idx = np.where(valid)[0]
        if idx.size < 2:
            return 1e9
        assigned = float(cost[ti, di])
        # best alternative excluding di
        alt = np.min(row[np.logical_and(valid, np.arange(row.shape[0]) != di)])
        return float(alt - assigned)

    def _col_margin(self, cost: np.ndarray, ti: int, di: int) -> float:
        col = cost[:, di]
        valid = col < float(self.cfg.valid_cost_thresh)
        idx = np.where(valid)[0]
        if idx.size < 2:
            return 1e9
        assigned = float(cost[ti, di])
        alt = np.min(col[np.logical_and(valid, np.arange(col.shape[0]) != ti)])
        return float(alt - assigned)

    def update(
        self,
        det_tlwh: np.ndarray,           # (N,4)
        det_scores: np.ndarray,         # (N,)
        det_feat: Optional[np.ndarray], # (N,D)
    ) -> List[Tuple[int, Tuple[float, float, float, float], float]]:
        self.frame_id += 1

        # predict
        for trk in self.tracks:
            trk.predict(self.kf)

        # filter detections
        keep = det_scores >= float(self.cfg.min_det_score)
        det_tlwh = det_tlwh[keep]
        det_scores = det_scores[keep]
        if det_feat is not None:
            det_feat = det_feat[keep]

        # associate
        cost = build_cost_matrix_v3(
            self.kf,
            self.tracks,
            det_tlwh,
            det_feat,
            w_app=float(self.cfg.w_app),
            w_iou=float(self.cfg.w_iou),
            w_motion=float(self.cfg.w_motion),
            iou_gate=float(self.cfg.iou_gate),
            maha_gate=float(self.cfg.maha_gate),
            maha_cost_scale=float(self.cfg.maha_cost_scale),
            app_gate=self.cfg.app_gate,
        )
        matches, unmatched_t, unmatched_d = hungarian_with_threshold(cost, float(self.cfg.match_thresh))

        amb_dets = self._ambiguous_dets_by_collision(det_tlwh)
        margin_thr = float(self.cfg.cost_margin) if self.cfg.freeze_on_margin else -1.0

        # update matched
        for ti, di in matches:
            trk = self.tracks[ti]
            tlwh = tuple(map(float, det_tlwh[di].tolist()))
            score = float(det_scores[di])
            feat = None if det_feat is None else det_feat[di]

            # default: update feature
            update_feat = True

            # 1) freeze if appearance change too large
            if (feat is not None) and (trk.feature is not None):
                app_dist, has_feat = appearance_distance(trk, feat)
                if has_feat and (app_dist > float(self.cfg.feature_update_gate)):
                    update_feat = False

            # 2) freeze on collision ambiguity
            if di in amb_dets:
                update_feat = False

            # 3) freeze on assignment ambiguity (small margin)
            if margin_thr > 0.0:
                rm = self._row_margin(cost, ti, di)
                cm = self._col_margin(cost, ti, di)
                if (rm < margin_thr) or (cm < margin_thr):
                    update_feat = False

            trk.update(
                self.kf, tlwh, score, feat,
                ema_alpha=self.cfg.ema_alpha,
                min_hits=self.cfg.min_hits,
                update_feature=update_feat,
                bank_size=self.cfg.feature_bank_size,
            )

        # create new for unmatched detections
        for di in unmatched_d:
            tlwh = tuple(map(float, det_tlwh[di].tolist()))
            score = float(det_scores[di])
            feat = None if det_feat is None else det_feat[di]
            self._create_track(tlwh, score, feat)

        # remove dead
        alive: List[Tracklet] = []
        for trk in self.tracks:
            if trk.time_since_update <= int(self.cfg.max_age):
                alive.append(trk)
        self.tracks = alive

        # output confirmed updated tracks
        outputs: List[Tuple[int, Tuple[float, float, float, float], float]] = []
        for trk in self.tracks:
            if trk.confirmed and trk.time_since_update == 0:
                outputs.append((trk.track_id, trk.to_tlwh(), float(trk.score)))
        return outputs
