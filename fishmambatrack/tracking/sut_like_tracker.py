
# fishmambatrack/tracking/sut_like_tracker.py
# -*- coding: utf-8 -*-
"""AMT association components used by the released AMT-L48 configuration."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple, Optional, Union

import numpy as np
from scipy.optimize import linear_sum_assignment


# -------------------------
# Box helpers
# -------------------------

def tlwh_to_xyxy(tlwh: np.ndarray) -> np.ndarray:
    """(N,4) tlwh -> xyxy"""
    out = tlwh.copy()
    out[:, 2] = out[:, 0] + out[:, 2]
    out[:, 3] = out[:, 1] + out[:, 3]
    return out


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Pairwise IoU between sets of boxes in xyxy.
    a: (N,4), b: (M,4) -> (N,M)
    """
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)

    inter_w = np.clip(inter_x2 - inter_x1, 0.0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0.0, None)
    inter = inter_w * inter_h

    area_a = np.clip((ax2 - ax1), 0.0, None) * np.clip((ay2 - ay1), 0.0, None)
    area_b = np.clip((bx2 - bx1), 0.0, None) * np.clip((by2 - by1), 0.0, None)

    union = area_a + area_b - inter + 1e-9
    return (inter / union).astype(np.float32)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Cosine similarity matrix between embeddings.
    a: (N,D), b: (M,D) -> (N,M)
    """
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    a = l2_normalize(a.astype(np.float32))
    b = l2_normalize(b.astype(np.float32))
    return (a @ b.T).astype(np.float32)


# -------------------------
# FishIoU (MFT25 / SU-T)
# -------------------------

@dataclass
class FishIoUParams:
    alpha: float = 0.15
    beta: float = 0.30
    gamma: float = 0.25
    mode: str = "fishiou"
    # Adaptive central box (shape-aware); disabled by default.
    adaptive_central: bool = False
    ar_ref: float = 2.0
    ar_scale_min: float = 0.6
    ar_scale_max: float = 1.4
    w1: float = 1.0
    w2: float = 0.3
    w3: float = 0.1
    w4: float = 0.2
    w5: float = 0.4
    eps: float = 1e-9


def _central_boxes(xyxy: np.ndarray, p: FishIoUParams) -> np.ndarray:
    """
    Central region boxes (Eq.16-17):
      Bc = [x1 + α w, y1 + β h, x2 - γ w, y2 - β h]
    """
    x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    w = np.clip(x2 - x1, 0.0, None)
    h = np.clip(y2 - y1, 0.0, None)

    if bool(getattr(p, "adaptive_central", False)):
        r = w / (h + p.eps)
        scale = np.clip(p.ar_ref / (r + p.eps), p.ar_scale_min, p.ar_scale_max)
        alpha = p.alpha * scale
        gamma = p.gamma * scale
        beta = p.beta
    else:
        alpha = p.alpha
        gamma = p.gamma
        beta = p.beta

    cx1 = x1 + alpha * w
    cy1 = y1 + beta * h
    cx2 = x2 - gamma * w
    cy2 = y2 - beta * h

    out = np.stack([cx1, cy1, cx2, cy2], axis=1).astype(np.float32)
    out[:, 2] = np.maximum(out[:, 2], out[:, 0] + 1e-3)
    out[:, 3] = np.maximum(out[:, 3], out[:, 1] + 1e-3)
    return out


def fishiou_matrix(
    det_tlwh: np.ndarray,
    trk_tlwh: np.ndarray,
    params: Optional[FishIoUParams] = None,
) -> np.ndarray:
    """
    Pairwise FishIoU between detection boxes and track boxes in tlwh.
    Returns similarity (D,T), higher is better.
    """
    if params is None:
        params = FishIoUParams()

    if det_tlwh.size == 0 or trk_tlwh.size == 0:
        return np.zeros((det_tlwh.shape[0], trk_tlwh.shape[0]), dtype=np.float32)

    b1 = tlwh_to_xyxy(det_tlwh.astype(np.float32))
    b2 = tlwh_to_xyxy(trk_tlwh.astype(np.float32))

    # IoU (Eq.14)
    iou = _iou_xyxy(b1, b2)

    # Center distance penalty dc (Eq.15)
    c1 = np.stack([(b1[:, 0] + b1[:, 2]) * 0.5, (b1[:, 1] + b1[:, 3]) * 0.5], axis=1)  # (D,2)
    c2 = np.stack([(b2[:, 0] + b2[:, 2]) * 0.5, (b2[:, 1] + b2[:, 3]) * 0.5], axis=1)  # (T,2)

    enc_x1 = np.minimum(b1[:, 0:1], b2[:, 0])
    enc_y1 = np.minimum(b1[:, 1:2], b2[:, 1])
    enc_x2 = np.maximum(b1[:, 2:3], b2[:, 2])
    enc_y2 = np.maximum(b1[:, 3:4], b2[:, 3])
    enc_w = np.clip(enc_x2 - enc_x1, 0.0, None)
    enc_h = np.clip(enc_y2 - enc_y1, 0.0, None)
    d2 = (enc_w ** 2 + enc_h ** 2) + params.eps

    dx = c1[:, 0:1] - c2[:, 0]
    dy = c1[:, 1:2] - c2[:, 1]
    dc = (dx ** 2 + dy ** 2) / d2

    mode = str(getattr(params, "mode", "fishiou") or "fishiou").lower()
    if mode in ("iou", "giou", "diou", "ciou"):
        if mode == "iou":
            return iou

        if mode == "giou":
            inter_x1 = np.maximum(b1[:, 0:1], b2[:, 0])
            inter_y1 = np.maximum(b1[:, 1:2], b2[:, 1])
            inter_x2 = np.minimum(b1[:, 2:3], b2[:, 2])
            inter_y2 = np.minimum(b1[:, 3:4], b2[:, 3])
            inter_w = np.clip(inter_x2 - inter_x1, 0.0, None)
            inter_h = np.clip(inter_y2 - inter_y1, 0.0, None)
            inter = inter_w * inter_h
            area_a = np.clip((b1[:, 2:3] - b1[:, 0:1]), 0.0, None) * np.clip((b1[:, 3:4] - b1[:, 1:2]), 0.0, None)
            area_b = np.clip((b2[:, 2] - b2[:, 0]), 0.0, None) * np.clip((b2[:, 3] - b2[:, 1]), 0.0, None)
            union = area_a + area_b - inter + params.eps
            enc_area = (enc_w * enc_h) + params.eps
            giou = iou - (enc_area - union) / enc_area
            return giou.astype(np.float32)

        diou = iou - dc
        if mode == "diou":
            return diou.astype(np.float32)

        # CIoU
        w1 = np.clip(b1[:, 2] - b1[:, 0], 0.0, None)
        h1 = np.clip(b1[:, 3] - b1[:, 1], 0.0, None)
        w2 = np.clip(b2[:, 2] - b2[:, 0], 0.0, None)
        h2 = np.clip(b2[:, 3] - b2[:, 1], 0.0, None)
        r1 = w1 / (h1 + params.eps)
        r2 = w2 / (h2 + params.eps)
        v = (4.0 / (np.pi ** 2)) * (np.arctan(r2)[None, :] - np.arctan(r1)[:, None]) ** 2
        alpha = v / (1.0 - iou + v + params.eps)
        ciou = iou - dc - alpha * v
        return ciou.astype(np.float32)

    # Central IoU cIoU (Eq.18)
    cb1 = _central_boxes(b1, params)
    cb2 = _central_boxes(b2, params)
    ciou = _iou_xyxy(cb1, cb2)

    # Aspect ratio consistency αr (Eq.19)
    w1 = np.clip(b1[:, 2] - b1[:, 0], 0.0, None)
    h1 = np.clip(b1[:, 3] - b1[:, 1], 0.0, None)
    w2 = np.clip(b2[:, 2] - b2[:, 0], 0.0, None)
    h2 = np.clip(b2[:, 3] - b2[:, 1], 0.0, None)
    r1 = w1 / (h1 + params.eps)
    r2 = w2 / (h2 + params.eps)
    ar = (np.minimum(r1[:, None], r2[None, :]) / (np.maximum(r1[:, None], r2[None, :]) + params.eps)).astype(np.float32)

    # Area ratio αa (Eq.20)
    a1 = w1 * h1
    a2 = w2 * h2
    aa = (np.minimum(a1[:, None], a2[None, :]) / (np.maximum(a1[:, None], a2[None, :]) + params.eps)).astype(np.float32)

    # Small-target scale factor s (Eq.21)
    s = (1.0 - np.exp(-np.minimum(a1[:, None], a2[None, :]) / 1000.0)).astype(np.float32)

    # FishIoU+ similarity. Clamp to [0, 1] so all association similarities
    # share the same bounded scale in the released fixed configuration.
    fiou = (
        params.w1 * iou
        + params.w2 * ciou
        + params.w3 * ar
        + params.w4 * aa
        - params.w5 * s * dc.astype(np.float32)
    ).astype(np.float32)

    return np.clip(fiou, 0.0, 1.0).astype(np.float32)


# -------------------------
# Tracker
# -------------------------

@dataclass
class SUTLikeTrackerConfig:
    # detection thresholds
    det_low_th: float = 0.10
    det_high_th: float = 0.60
    det_nms_iou: Optional[float] = None  # optional NMS on per-frame dets (IoU threshold)

    # association thresholds
    fishiou_th: float = 0.25  # τiou
    fishiou_th_low: Optional[float] = None  # Stage-2 (low-conf) FishIoU gate; None -> use fishiou_th
    # Stage-2 dynamic FishIoU gate based on detection score:
    # gate = lerp(max_gate -> min_gate) as score goes from det_low_th -> det_high_th
    fishiou_th_low_min: Optional[float] = None
    fishiou_th_low_max: Optional[float] = None

    # stage weights
    w_fishiou: float = 1.0
    fishiou_axis_w: float = 0.0  # add axis similarity into FishIoU (Axis-Consistent FishIoU)
    w_app: float = 1.0
    w_app_low: float = 0.5
    w_app_stage3: float = 0.0  # appearance weight for Stage-3 (last chance) matching
    # Optional: dynamic Stage-2 appearance weight based on detection score.
    # weight = lerp(w_app_low_min -> w_app_low_max) as score goes from det_low_th -> det_high_th.
    w_app_low_min: Optional[float] = None
    w_app_low_max: Optional[float] = None
    w_axis: float = 0.0  # small axis-distance penalty if Detection/Track provide axis
    axis_use_abs: bool = True  # if True, treat axis as undirected (abs dot); if False, keep direction
    # Optional motion penalty on center distance (scale-normalized).
    w_motion: float = 0.0
    motion_scale: float = 1.0  # dist_norm /= motion_scale before clamping to [0,1]
    motion_gate: Optional[float] = None  # if set, reject pairs with dist_norm_raw > gate
    # If False, disable appearance/ReID usage (no cosine sim, no long-term reid).
    use_reid: bool = True
    # Optional crowd-aware appearance damping.
    # A detection is considered "crowded" if it overlaps >= crowd_count_th tracks with FishIoU >= crowd_fishiou_th.
    # If set, crowded detections use min(current_app_weight, w_app_crowd) during matching.
    w_app_crowd: Optional[float] = None
    crowd_fishiou_th: float = 0.05
    crowd_count_th: int = 2

    # stage behavior
    stage2_use_app: bool = True  # low-conf stage uses appearance or not
    stage2_update_emb: bool = False  # low-conf matches update embedding/axis or not
    stage3_update_emb: bool = True   # last-chance (IoU-only) matches update embedding/axis or not
    lt_update_emb: bool = True       # long-term correction matches update embedding/axis or not
    stage2_refine_app: bool = False  # optional second pass on low-conf using appearance-only
    stage2_refine_sim_th: float = 0.0
    stage2_refine_fishiou_gate: float = 0.0
    # If True, do NOT update embedding/axis when the matched detection is "crowded" (bbox/velocity still update).
    freeze_emb_in_crowd: bool = False

    # embedding/axis update gating (for all stages; applied on top of stage*_update_emb)
    emb_update_sim_th: float = 0.0       # if >0, require raw cosine sim >= this to update emb/axis
    emb_update_fishiou_th: float = 0.0   # if >0, require FishIoU >= this to update emb/axis

    # long-term correction
    reid_long_th: float = 0.40
    reid_long_fishiou_gate: float = 0.05  # gate long-term correction by FishIoU to avoid crazy matches
    emb_gain_high: float = 1.3
    emb_gain_low: float = 1.2

    # track management
    max_age: int = 30
    min_hits: int = 1
    use_confirmed_cascade: bool = False  # BoT-SORT style confirmed/unconfirmed cascade
    drop_unconfirmed_on_miss: bool = False
    # If set and min_hits > 1: allow high-score tracks to be output before confirmation.
    # This reduces the FN penalty of min_hits while still filtering low-score one-frame false positives.
    min_hits_score_gate: Optional[float] = None

    # embedding update
    emb_bank_size: int = 1   # number of embeddings stored per track (max-sim over bank)
    emb_momentum: float = 1.0  # baseline_app
    inertia: float = 0.90      # old-velocity weight in v = (1-inertia)*delta + inertia*v
    # Robust bank similarity aggregation:
    # final_sim = (1-alpha)*max_sim + alpha*mean_topk_sim.
    # alpha=0 keeps legacy max-sim behavior.
    emb_bank_consensus_alpha: float = 0.0
    # top-k for consensus mean (<=0 means use all bank entries).
    emb_bank_consensus_topk: int = 0
    # Similarity source for emb-update gate:
    # True -> compare det to bank aggregation; False -> compare to current track prototype only.
    emb_update_sim_use_bank: bool = True
    # Optional diversity-aware bank write for bank_size>1:
    # if max cosine(new_emb, bank) >= threshold, skip appending this embedding.
    # Set >1 to disable (default).
    emb_bank_diversity_th: float = 1.01

    # similarity post-processing
    sim_relu: bool = False
    fishiou_params: Optional[FishIoUParams] = None
    # Optional switch for disabling the cascade path.
    association_mode: str = "cascade"  # cascade | single_stage


@dataclass
class Detection:
    tlwh: np.ndarray
    score: float
    emb: Optional[np.ndarray] = None
    axis: Optional[np.ndarray] = None


@dataclass
class Track:
    track_id: int
    tlwh: np.ndarray
    score: float
    emb: Optional[np.ndarray] = None
    emb_bank: Optional[Deque[np.ndarray]] = None
    axis: Optional[np.ndarray] = None
    axis_bank: Optional[Deque[np.ndarray]] = None

    age: int = 1
    hits: int = 1
    time_since_update: int = 0

    last_observation: Optional[np.ndarray] = None
    velocity: Optional[np.ndarray] = None  # (dx, dy)

    def predict(self, cfg: SUTLikeTrackerConfig) -> np.ndarray:
        """
        Predict tlwh for this frame.
        This MUST be called exactly once per frame for each active track.
        """
        self.age += 1
        self.time_since_update += 1

        if self.velocity is None:
            return self.tlwh.copy()

        pred = self.tlwh.copy()
        pred[0] += float(self.velocity[0])
        pred[1] += float(self.velocity[1])
        return pred

    def update(self, det: Detection, cfg: SUTLikeTrackerConfig, *, update_emb: bool = True) -> None:
        self.time_since_update = 0
        self.hits += 1
        self.score = float(det.score)

        prev = self.tlwh.copy()
        self.tlwh = det.tlwh.copy()
        self.last_observation = det.tlwh.copy()

        # velocity update (center displacement) with inertia coefficient
        prev_c = prev[:2] + prev[2:] * 0.5
        cur_c = det.tlwh[:2] + det.tlwh[2:] * 0.5
        delta = (cur_c - prev_c).astype(np.float32)

        if self.velocity is None:
            self.velocity = delta
        else:
            mu = float(cfg.inertia)
            # mostly follow latest delta, keep a small fraction of previous velocity
            self.velocity = (1.0 - mu) * delta + mu * self.velocity

        # embedding update
        if update_emb and det.emb is not None:
            det_emb = det.emb.astype(np.float32)

            if self.emb is None:
                self.emb = det_emb
            else:
                m = float(cfg.emb_momentum)
                if m >= 1.0:
                    self.emb = det_emb
                else:
                    self.emb = (1.0 - m) * self.emb + m * det_emb

            # keep a small embedding bank to handle large appearance changes
            bank_size = int(getattr(cfg, "emb_bank_size", 1))
            if bank_size > 0:
                if self.emb_bank is None or self.emb_bank.maxlen != bank_size:
                    prev = list(self.emb_bank) if self.emb_bank else []
                    self.emb_bank = deque(prev[-bank_size:], maxlen=bank_size)
                det_emb_n = l2_normalize(det_emb.reshape(1, -1))[0]
                push_bank = True
                if bank_size > 1 and self.emb_bank:
                    div_th = float(getattr(cfg, "emb_bank_diversity_th", 1.01))
                    if div_th <= 1.0:
                        bank_now = np.stack(list(self.emb_bank), axis=0).astype(np.float32)
                        bank_now = l2_normalize(bank_now)
                        sim_max = float((bank_now @ det_emb_n.reshape(-1, 1)).max())
                        if sim_max >= div_th:
                            push_bank = False
                if push_bank:
                    self.emb_bank.append(det_emb_n)

        # axis update (same gating as embedding; Stage-2 should not pollute it)
        if update_emb and det.axis is not None:
            det_axis = det.axis.astype(np.float32)
            det_axis = l2_normalize(det_axis.reshape(1, -1))[0]
            self.axis = det_axis

            bank_size = int(getattr(cfg, "emb_bank_size", 1))
            if bank_size > 0:
                if self.axis_bank is None or self.axis_bank.maxlen != bank_size:
                    prev = list(self.axis_bank) if self.axis_bank else []
                    self.axis_bank = deque(prev[-bank_size:], maxlen=bank_size)
                self.axis_bank.append(det_axis)


class SUTLikeTracker:
    """
    Multi-level cascade tracker (Algorithm 1 in SU-T paper).
    """
    def __init__(self, cfg: SUTLikeTrackerConfig, fishiou_params: Optional[FishIoUParams] = None):
        self.cfg = cfg
        cfg_fishiou = getattr(cfg, "fishiou_params", None)
        self.fishiou_params = cfg_fishiou or fishiou_params or FishIoUParams()
        self.tracks: List[Track] = []
        self._next_id = 1
        self._frame_id: int = 0
        self.diagnostics: Dict[str, int] = {}
        self.diagnostic_events: List[Dict[str, object]] = []
        self.reset_diagnostics()

    def reset(self) -> None:
        self.tracks = []
        self._next_id = 1
        self._frame_id = 0
        self.reset_diagnostics()

    def reset_diagnostics(self) -> None:
        keys = [
            "feature_update_events",
            "feature_update_gate_pass",
            "geometry_gate_rejects",
            "appearance_gate_rejects",
            "crowd_gate_rejects",
            "stage_update_disabled",
            "actual_history_writes",
            "matched_history_writes",
            "new_track_history_inits",
            "reactivation_pairs",
            "reactivation_attempts",
            "reactivation_matches",
            "reactivation_history_writes",
        ]
        self.diagnostics = {k: 0 for k in keys}
        self.diagnostic_events = []

    def _diag_inc(self, key: str, value: int = 1) -> None:
        self.diagnostics[key] = int(self.diagnostics.get(key, 0)) + int(value)

    def _record_event(
        self,
        *,
        stage: str,
        track_id: int,
        det: Detection,
        update_emb: bool,
        reason: str,
        fishiou: Optional[float] = None,
        raw_sim: Optional[float] = None,
    ) -> None:
        self.diagnostic_events.append({
            "frame": int(self._frame_id),
            "stage": str(stage),
            "track_id": int(track_id),
            "score": float(det.score),
            "tlwh": det.tlwh.astype(float).tolist(),
            "update_emb": bool(update_emb),
            "reason": str(reason),
            "fishiou": None if fishiou is None else float(fishiou),
            "raw_sim": None if raw_sim is None else float(raw_sim),
        })

    def _prune(self) -> None:
        alive: List[Track] = []
        for t in self.tracks:
            if t.time_since_update > self.cfg.max_age:
                continue
            if bool(getattr(self.cfg, "drop_unconfirmed_on_miss", False)):
                if t.hits < self.cfg.min_hits and t.time_since_update > 0:
                    continue
            alive.append(t)
        self.tracks = alive

    def _cos_sim(self, dets: List[Detection], trks: List[Track]) -> Optional[np.ndarray]:
        if not bool(getattr(self.cfg, "use_reid", True)):
            return None
        if not dets or not trks:
            return None
        if any(d.emb is None for d in dets):
            return None
        det_embs = l2_normalize(np.stack([d.emb for d in dets], axis=0).astype(np.float32))

        D, T = len(dets), len(trks)
        sim = np.zeros((D, T), dtype=np.float32)
        for j, t in enumerate(trks):
            if t.emb_bank:
                bank = np.stack(list(t.emb_bank), axis=0).astype(np.float32)
                bank = l2_normalize(bank)
                sim_bank = det_embs @ bank.T
                sim[:, j] = self._aggregate_bank_sim(sim_bank)
            else:
                if t.emb is None:
                    return None
                emb = l2_normalize(t.emb.astype(np.float32).reshape(1, -1))
                sim[:, j] = (det_embs @ emb.T)[:, 0]

        if bool(getattr(self.cfg, "sim_relu", False)):
            sim = np.maximum(sim, 0.0)

        gains = np.array([
            (self.cfg.emb_gain_high if d.score >= self.cfg.det_high_th else self.cfg.emb_gain_low)
            for d in dets
        ], dtype=np.float32)[:, None]
        return sim * gains

    def _cos_sim_pair_raw(self, det: Detection, trk: Track, *, use_bank: bool = True) -> Optional[float]:
        """
        Raw cosine similarity for a single det-track pair (no gain scaling).
        Uses track embedding bank if present.
        """
        if not bool(getattr(self.cfg, "use_reid", True)):
            return None
        if det.emb is None:
            return None

        d = det.emb.astype(np.float32).reshape(1, -1)
        d = l2_normalize(d)[0]

        if use_bank and trk.emb_bank:
            bank = np.stack(list(trk.emb_bank), axis=0).astype(np.float32)
            bank = l2_normalize(bank)
            sim_bank = (bank @ d.reshape(-1, 1)).reshape(1, -1).astype(np.float32, copy=False)
            return float(self._aggregate_bank_sim(sim_bank)[0])

        if trk.emb is None:
            return None
        t = trk.emb.astype(np.float32).reshape(1, -1)
        t = l2_normalize(t)[0]
        return float(np.dot(d, t))

    def _aggregate_bank_sim(self, sim_bank: np.ndarray) -> np.ndarray:
        """
        Aggregate per-track bank similarities for each detection.
        sim_bank: (D, B)
        """
        if sim_bank.ndim != 2 or sim_bank.shape[1] == 0:
            return np.zeros((sim_bank.shape[0],), dtype=np.float32)

        max_sim = sim_bank.max(axis=1).astype(np.float32, copy=False)
        if sim_bank.shape[1] == 1:
            return max_sim

        alpha = float(getattr(self.cfg, "emb_bank_consensus_alpha", 0.0))
        if alpha <= 0.0:
            return max_sim
        if alpha >= 1.0:
            alpha = 1.0

        topk = int(getattr(self.cfg, "emb_bank_consensus_topk", 0))
        bsz = int(sim_bank.shape[1])
        if topk <= 0 or topk >= bsz:
            mean_sim = sim_bank.mean(axis=1).astype(np.float32, copy=False)
        else:
            # take top-k values per row in O(B) using partition
            kth = bsz - topk
            topk_vals = np.partition(sim_bank, kth=kth, axis=1)[:, kth:]
            mean_sim = topk_vals.mean(axis=1).astype(np.float32, copy=False)

        return ((1.0 - alpha) * max_sim + alpha * mean_sim).astype(np.float32, copy=False)

    def _crowd_mask_from_fishiou(self, fiou: np.ndarray) -> np.ndarray:
        """
        Return a boolean mask (D,) indicating detections that overlap multiple tracks.
        Uses config: crowd_fishiou_th / crowd_count_th.
        """
        if fiou.ndim != 2:
            return np.zeros((0,), dtype=np.bool_)
        D = int(fiou.shape[0])
        if D == 0:
            return np.zeros((0,), dtype=np.bool_)

        th = float(getattr(self.cfg, "crowd_fishiou_th", 0.0))
        k = int(getattr(self.cfg, "crowd_count_th", 0))
        if th <= 0.0 or k <= 1:
            return np.zeros((D,), dtype=np.bool_)

        return (fiou >= th).sum(axis=1) >= k

    def _apply_crowd_app_weight(self, app_weight: Union[float, np.ndarray], fiou: np.ndarray) -> Union[float, np.ndarray]:
        """
        Optionally damp appearance weight for "crowded" detections, using:
          w_app_crowd, crowd_fishiou_th, crowd_count_th
        """
        w_crowd = getattr(self.cfg, "w_app_crowd", None)
        if w_crowd is None:
            return app_weight

        try:
            w_crowd_f = float(w_crowd)
        except Exception:
            return app_weight
        if w_crowd_f < 0.0 or fiou.ndim != 2:
            return app_weight

        crowd = self._crowd_mask_from_fishiou(fiou)
        if crowd.size == 0 or not bool(crowd.any()):
            return app_weight

        if isinstance(app_weight, np.ndarray):
            w = app_weight.astype(np.float32, copy=True).reshape(-1)
            if w.shape[0] != int(fiou.shape[0]):
                return app_weight
            w[crowd] = np.minimum(w[crowd], w_crowd_f)
            return w

        base = float(app_weight)
        if base <= 0.0:
            return app_weight
        w = np.full((int(fiou.shape[0]),), base, dtype=np.float32)
        w[crowd] = min(base, w_crowd_f)
        return w

    def _should_update_emb_reason(self, det: Detection, trk: Track, *, fishiou: float) -> Tuple[bool, str, Optional[float]]:
        """
        Confidence gate for updating embedding/axis (bbox/velocity always update).
        """
        if not bool(getattr(self.cfg, "use_reid", True)):
            return False, "reid_disabled", None
        fiou_th = float(getattr(self.cfg, "emb_update_fishiou_th", 0.0))
        if fiou_th > 0.0 and float(fishiou) < fiou_th:
            return False, "geometry_gate", None

        sim_th = float(getattr(self.cfg, "emb_update_sim_th", 0.0))
        raw_sim: Optional[float] = None
        if sim_th > 0.0:
            use_bank = bool(getattr(self.cfg, "emb_update_sim_use_bank", True))
            raw_sim = self._cos_sim_pair_raw(det, trk, use_bank=use_bank)
            if raw_sim is None or float(raw_sim) < sim_th:
                return False, "appearance_gate", raw_sim
        return True, "allowed", raw_sim

    def _should_update_emb(self, det: Detection, trk: Track, *, fishiou: float) -> bool:
        ok, _, _ = self._should_update_emb_reason(det, trk, fishiou=fishiou)
        return bool(ok)

    def _decide_feature_update(
        self,
        *,
        stage: str,
        det: Detection,
        trk: Track,
        fishiou: float,
        stage_allows_update: bool,
        crowd_block: bool = False,
        apply_quality_gate: bool = True,
    ) -> bool:
        raw_sim: Optional[float] = None
        gate_ok = True
        reason = "allowed"

        if not bool(stage_allows_update):
            self._diag_inc("stage_update_disabled")
            gate_ok = False
            reason = "stage_disabled"
        elif apply_quality_gate:
            self._diag_inc("feature_update_events")
            gate_ok, reason, raw_sim = self._should_update_emb_reason(det, trk, fishiou=float(fishiou))
            if gate_ok:
                self._diag_inc("feature_update_gate_pass")
            elif reason == "geometry_gate":
                self._diag_inc("geometry_gate_rejects")
            elif reason == "appearance_gate":
                self._diag_inc("appearance_gate_rejects")
        else:
            if bool(getattr(self.cfg, "use_reid", True)):
                raw_sim = self._cos_sim_pair_raw(det, trk, use_bank=True)

        update_emb = bool(stage_allows_update and gate_ok)
        if update_emb and bool(crowd_block):
            self._diag_inc("crowd_gate_rejects")
            update_emb = False
            reason = "crowd_gate"

        if update_emb:
            self._diag_inc("actual_history_writes")
            self._diag_inc("matched_history_writes")
            if str(stage) == "reactivation":
                self._diag_inc("reactivation_history_writes")

        self._record_event(
            stage=stage,
            track_id=int(trk.track_id),
            det=det,
            update_emb=bool(update_emb),
            reason=reason,
            fishiou=float(fishiou),
            raw_sim=raw_sim,
        )
        return bool(update_emb)

    @staticmethod
    def _nms_dets(dets: List[Detection], iou_th: float) -> List[Detection]:
        """
        Simple per-frame NMS on tlwh boxes (class-agnostic).
        Keeps higher-score dets when IoU > iou_th.
        """
        if len(dets) <= 1:
            return dets
        th = float(iou_th)
        if not (0.0 < th < 1.0):
            return dets

        boxes = np.stack([d.tlwh for d in dets], axis=0).astype(np.float32)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
        scores = np.array([float(d.score) for d in dets], dtype=np.float32)

        order = scores.argsort()[::-1]
        keep: List[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            ious = _iou_xyxy(xyxy[i : i + 1], xyxy[rest])[0]
            order = rest[ious <= th]

        return [dets[i] for i in keep]

    def _hungarian(
        self,
        dets: List[Detection],
        trks: List[Track],
        trk_boxes_tlwh: np.ndarray,
        app_weight: Union[float, np.ndarray],
        fishiou_gate: Union[float, np.ndarray],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int], np.ndarray]:
        """
        One round matching: maximize S = w_fishiou*FishIoU + app_weight*sim.
        Gate by FishIoU >= fishiou_gate.
        Returns matches, unmatched det indices, unmatched trk indices, fishiou matrix.
        """
        D, T = len(dets), len(trks)
        if D == 0 or T == 0:
            return [], list(range(D)), list(range(T)), np.zeros((D, T), dtype=np.float32)

        det_boxes = np.stack([d.tlwh for d in dets], axis=0).astype(np.float32)
        fiou = fishiou_matrix(det_boxes, trk_boxes_tlwh.astype(np.float32), self.fishiou_params)

        # Optional crowd-aware damping of appearance weight (dense / collision scenes).
        app_weight = self._apply_crowd_app_weight(app_weight, fiou)

        # Optional: add axis similarity into FishIoU (Axis-Consistent FishIoU).
        fishiou_axis_w = float(getattr(self.cfg, "fishiou_axis_w", 0.0))
        if fishiou_axis_w > 0.0:
            det_has_axis = np.array([getattr(d, "axis", None) is not None for d in dets], dtype=np.bool_)
            if bool(det_has_axis.any()):
                det_axes = np.stack([d.axis for d in dets if d.axis is not None], axis=0).astype(np.float32)
                det_axes = l2_normalize(det_axes)
                det_idx = np.where(det_has_axis)[0]
                axis_use_abs = bool(getattr(self.cfg, "axis_use_abs", True))

                axis_sim = np.zeros((D, T), dtype=np.float32)
                for j, t in enumerate(trks):
                    bank = None
                    if getattr(t, "axis_bank", None):
                        bank = np.stack(list(t.axis_bank), axis=0).astype(np.float32)
                    elif getattr(t, "axis", None) is not None:
                        bank = np.stack([t.axis], axis=0).astype(np.float32)
                    if bank is None or bank.size == 0:
                        continue
                    bank = l2_normalize(bank)
                    sim_mat = det_axes @ bank.T
                    if axis_use_abs:
                        sim_mat = np.abs(sim_mat)
                    else:
                        sim_mat = 0.5 * (sim_mat + 1.0)
                    sim_best = sim_mat.max(axis=1)
                    axis_sim[det_idx, j] = np.clip(sim_best, 0.0, 1.0).astype(np.float32)

                fiou = fiou + fishiou_axis_w * axis_sim

        S = self.cfg.w_fishiou * fiou
        use_app = False
        w: Optional[np.ndarray] = None
        if isinstance(app_weight, np.ndarray):
            w = app_weight.astype(np.float32, copy=False).reshape(-1)
            if w.shape[0] != D:
                raise ValueError(f"app_weight array must be (D,), got {tuple(w.shape)} D={D}")
            use_app = bool(float(w.max()) > 0.0)
        else:
            use_app = bool(float(app_weight) > 0.0)

        if use_app:
            sim = self._cos_sim(dets, trks)
            if sim is not None:
                if w is not None:
                    S = S + w[:, None] * sim
                else:
                    S = S + float(app_weight) * sim

        cost = -S.astype(np.float32)

        # Optional: add a small axis distance term to the cost matrix.
        w_axis = float(getattr(self.cfg, "w_axis", 0.0))
        if w_axis > 0.0:
            det_has_axis = np.array([getattr(d, "axis", None) is not None for d in dets], dtype=np.bool_)
            if bool(det_has_axis.any()):
                det_axes = np.stack([d.axis for d in dets if d.axis is not None], axis=0).astype(np.float32)
                det_axes = l2_normalize(det_axes)
                axis_use_abs = bool(getattr(self.cfg, "axis_use_abs", True))

                axis_dist = np.zeros((D, T), dtype=np.float32)
                for j, t in enumerate(trks):
                    bank = None
                    if getattr(t, "axis_bank", None):
                        bank = np.stack(list(t.axis_bank), axis=0).astype(np.float32)
                    elif getattr(t, "axis", None) is not None:
                        bank = np.stack([t.axis], axis=0).astype(np.float32)
                    if bank is None or bank.size == 0:
                        continue
                    bank = l2_normalize(bank)
                    sim_mat = det_axes @ bank.T
                    if axis_use_abs:
                        sim_mat = np.abs(sim_mat)
                    sim_best = sim_mat.max(axis=1)
                    sim_best = np.clip(sim_best, -1.0, 1.0)
                    axis_dist[det_has_axis, j] = (0.5 * (1.0 - sim_best)).astype(np.float32)

                cost = cost + w_axis * axis_dist

        # Optional: add a motion penalty based on center distance (scale-normalized).
        w_motion = float(getattr(self.cfg, "w_motion", 0.0))
        if w_motion > 0.0:
            det_xy = det_boxes[:, :2] + 0.5 * det_boxes[:, 2:]
            trk_xy = trk_boxes_tlwh[:, :2] + 0.5 * trk_boxes_tlwh[:, 2:]
            dx = det_xy[:, None, 0] - trk_xy[None, :, 0]
            dy = det_xy[:, None, 1] - trk_xy[None, :, 1]
            dist = np.sqrt(dx * dx + dy * dy).astype(np.float32)
            det_area = np.clip(det_boxes[:, 2] * det_boxes[:, 3], 1.0, None).astype(np.float32)
            scale = np.sqrt(det_area)[:, None]
            dist_raw = dist / (scale + 1e-6)
            motion_scale = float(getattr(self.cfg, "motion_scale", 1.0))
            dist_norm = dist_raw
            if motion_scale > 1e-6:
                dist_norm = dist_norm / motion_scale
            dist_norm = np.clip(dist_norm, 0.0, 1.0)
            cost = cost + w_motion * dist_norm.astype(np.float32)

            motion_gate = getattr(self.cfg, "motion_gate", None)
            if motion_gate is not None and float(motion_gate) > 0.0:
                cost[dist_raw > float(motion_gate)] = 1e6
        if isinstance(fishiou_gate, np.ndarray):
            gate = fishiou_gate.astype(np.float32, copy=False).reshape(-1)
            if gate.shape[0] != D:
                raise ValueError(f"fishiou_gate array must be (D,), got {tuple(gate.shape)} D={D}")
            cost[fiou < gate[:, None]] = 1e6
        else:
            cost[fiou < float(fishiou_gate)] = 1e6

        r, c = linear_sum_assignment(cost)
        matches = []
        unmatched_d = set(range(D))
        unmatched_t = set(range(T))
        for ri, ci in zip(r.tolist(), c.tolist()):
            if cost[ri, ci] >= 1e5:
                continue
            matches.append((ri, ci))
            unmatched_d.discard(ri)
            unmatched_t.discard(ci)

        return matches, sorted(unmatched_d), sorted(unmatched_t), fiou

    def _long_term_correction(
        self,
        dets: List[Detection],
        trks: List[Track],
        trk_boxes_tlwh: np.ndarray,
        fishiou_gate: float = 0.05,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int], np.ndarray]:
        """
        Greedy reid correction (sim >= reid_long_th).
        Optionally gate by FishIoU >= fishiou_gate to avoid crazy matches.
        """
        D, T = len(dets), len(trks)
        if D == 0 or T == 0:
            return [], list(range(D)), list(range(T)), np.zeros((D, T), dtype=np.float32)

        sim = self._cos_sim(dets, trks)
        if sim is None:
            return [], list(range(D)), list(range(T)), np.zeros((D, T), dtype=np.float32)

        det_boxes = np.stack([d.tlwh for d in dets], axis=0).astype(np.float32)
        fiou = fishiou_matrix(det_boxes, trk_boxes_tlwh.astype(np.float32), self.fishiou_params)
        sim = sim.copy()
        sim[fiou < fishiou_gate] = -1.0
        valid = sim >= float(self.cfg.reid_long_th)
        n_valid = int(valid.sum())
        self._diag_inc("reactivation_pairs", n_valid)
        self._diag_inc("reactivation_attempts", n_valid)

        cand = [(float(sim[i, j]), i, j) for i in range(D) for j in range(T)]
        cand.sort(reverse=True, key=lambda x: x[0])

        used_d, used_t = set(), set()
        matches = []
        for s, i, j in cand:
            if s < self.cfg.reid_long_th:
                break
            if i in used_d or j in used_t:
                continue
            used_d.add(i)
            used_t.add(j)
            matches.append((i, j))
        self._diag_inc("reactivation_matches", len(matches))

        ud = [i for i in range(D) if i not in used_d]
        ut = [j for j in range(T) if j not in used_t]
        return matches, ud, ut, fiou

    def _update_confirmed_cascade(
        self,
        dets_high: List[Detection],
        dets_low: List[Detection],
        trks: List[Track],
        pred_boxes: np.ndarray,
    ) -> List[Track]:
        cfg = self.cfg
        confirmed_idx = [i for i, t in enumerate(trks) if t.hits >= cfg.min_hits]
        unconfirmed_idx = [i for i, t in enumerate(trks) if t.hits < cfg.min_hits]

        confirmed_trks = [trks[i] for i in confirmed_idx]
        unconfirmed_trks = [trks[i] for i in unconfirmed_idx]
        pred_confirmed = pred_boxes[confirmed_idx] if confirmed_idx else np.zeros((0, 4), dtype=np.float32)
        pred_unconfirmed = pred_boxes[unconfirmed_idx] if unconfirmed_idx else np.zeros((0, 4), dtype=np.float32)

        # Stage-1: confirmed tracks <-> high-conf detections (FishIoU + appearance)
        m1, ud1, ut1, fiou1 = self._hungarian(
            dets_high, confirmed_trks,
            trk_boxes_tlwh=pred_confirmed,
            app_weight=cfg.w_app,
            fishiou_gate=cfg.fishiou_th,
        )
        crowd1: Optional[np.ndarray] = None
        if bool(getattr(cfg, "freeze_emb_in_crowd", False)) and fiou1.size > 0:
            crowd1 = self._crowd_mask_from_fishiou(fiou1)
        for di, tj in m1:
            update_emb = self._decide_feature_update(
                stage="stage1_confirmed",
                det=dets_high[di],
                trk=confirmed_trks[tj],
                fishiou=float(fiou1[di, tj]),
                stage_allows_update=True,
                crowd_block=bool(crowd1 is not None and bool(crowd1[di])),
            )
            confirmed_trks[tj].update(dets_high[di], cfg, update_emb=bool(update_emb))

        rem_confirmed_idx1 = [confirmed_idx[j] for j in ut1]
        rem_confirmed_trks1 = [trks[j] for j in rem_confirmed_idx1]
        rem_pred1 = pred_confirmed[ut1] if ut1 else np.zeros((0, 4), dtype=np.float32)
        dets_high_rem = [dets_high[i] for i in ud1]

        # Stage-1b: unconfirmed tracks <-> remaining high-conf detections (FishIoU only)
        m1b, ud1b, ut1b, fiou1b = self._hungarian(
            dets_high_rem, unconfirmed_trks,
            trk_boxes_tlwh=pred_unconfirmed,
            app_weight=0.0,
            fishiou_gate=cfg.fishiou_th,
        )
        crowd1b: Optional[np.ndarray] = None
        if bool(getattr(cfg, "freeze_emb_in_crowd", False)) and fiou1b.size > 0:
            crowd1b = self._crowd_mask_from_fishiou(fiou1b)
        for di, tj in m1b:
            update_emb = self._decide_feature_update(
                stage="stage1_unconfirmed",
                det=dets_high_rem[di],
                trk=unconfirmed_trks[tj],
                fishiou=float(fiou1b[di, tj]),
                stage_allows_update=True,
                crowd_block=bool(crowd1b is not None and bool(crowd1b[di])),
            )
            unconfirmed_trks[tj].update(dets_high_rem[di], cfg, update_emb=bool(update_emb))

        dets_high_rem2 = [dets_high_rem[i] for i in ud1b]

        # Stage-2: low-conf detections <-> remaining confirmed tracks
        stage2_gate: Union[float, np.ndarray] = float(cfg.fishiou_th)
        if getattr(cfg, "fishiou_th_low_min", None) is not None and getattr(cfg, "fishiou_th_low_max", None) is not None:
            g_min = float(cfg.fishiou_th_low_min)
            g_max = float(cfg.fishiou_th_low_max)
            if g_min > g_max:
                g_min, g_max = g_max, g_min
            s_lo = float(cfg.det_low_th)
            s_hi = float(cfg.det_high_th)
            den = max(1e-6, s_hi - s_lo)
            scores = np.array([float(d.score) for d in dets_low], dtype=np.float32)
            t = np.clip((scores - s_lo) / den, 0.0, 1.0)
            stage2_gate = (g_max - t * (g_max - g_min)).astype(np.float32)
        elif getattr(cfg, "fishiou_th_low", None) is not None:
            stage2_gate = float(cfg.fishiou_th_low)

        if bool(getattr(cfg, "stage2_use_app", True)):
            stage2_app: Union[float, np.ndarray] = float(cfg.w_app_low)
            if getattr(cfg, "w_app_low_min", None) is not None and getattr(cfg, "w_app_low_max", None) is not None:
                w_min = float(cfg.w_app_low_min)
                w_max = float(cfg.w_app_low_max)
                if w_min > w_max:
                    w_min, w_max = w_max, w_min
                s_lo = float(cfg.det_low_th)
                s_hi = float(cfg.det_high_th)
                den = max(1e-6, s_hi - s_lo)
                scores = np.array([float(d.score) for d in dets_low], dtype=np.float32)
                t = np.clip((scores - s_lo) / den, 0.0, 1.0)
                stage2_app = (w_min + t * (w_max - w_min)).astype(np.float32)
        else:
            stage2_app = 0.0

        m2, ud2, ut2, fiou2 = self._hungarian(
            dets_low, rem_confirmed_trks1,
            trk_boxes_tlwh=rem_pred1,
            app_weight=stage2_app,
            fishiou_gate=stage2_gate,
        )
        crowd2: Optional[np.ndarray] = None
        if bool(getattr(cfg, "freeze_emb_in_crowd", False)) and fiou2.size > 0:
            crowd2 = self._crowd_mask_from_fishiou(fiou2)
        for di, tj in m2:
            update_emb = self._decide_feature_update(
                stage="stage2_low",
                det=dets_low[di],
                trk=rem_confirmed_trks1[tj],
                fishiou=float(fiou2[di, tj]),
                stage_allows_update=bool(cfg.stage2_update_emb),
                crowd_block=bool(crowd2 is not None and bool(crowd2[di])),
            )
            rem_confirmed_trks1[tj].update(dets_low[di], cfg, update_emb=bool(update_emb))

        rem_confirmed_idx2 = [rem_confirmed_idx1[j] for j in ut2]
        rem_confirmed_trks2 = [trks[j] for j in rem_confirmed_idx2]
        rem_pred2 = rem_pred1[ut2] if ut2 else np.zeros((0, 4), dtype=np.float32)
        rem_low = [dets_low[i] for i in ud2]

        # Optional: refine low-conf matches using appearance-only with a high sim threshold
        if bool(getattr(cfg, "stage2_refine_app", False)) and rem_low and rem_confirmed_trks2:
            sim = self._cos_sim(rem_low, rem_confirmed_trks2)
            if sim is not None:
                det_boxes = np.stack([d.tlwh for d in rem_low], axis=0).astype(np.float32)
                fiou = fishiou_matrix(det_boxes, rem_pred2.astype(np.float32), self.fishiou_params)
                cost = -sim.astype(np.float32)
                sim_th = float(getattr(cfg, "stage2_refine_sim_th", 0.0))
                if sim_th > 0.0:
                    cost[sim < sim_th] = 1e6
                fiou_gate = float(getattr(cfg, "stage2_refine_fishiou_gate", 0.0))
                if fiou_gate > 0.0:
                    cost[fiou < fiou_gate] = 1e6

                r, c = linear_sum_assignment(cost)
                matched_d = set()
                matched_t = set()
                for ri, ci in zip(r.tolist(), c.tolist()):
                    if cost[ri, ci] >= 1e5:
                        continue
                    det = rem_low[int(ri)]
                    trk = rem_confirmed_trks2[int(ci)]
                    update_emb = self._decide_feature_update(
                        stage="stage2_refine",
                        det=det,
                        trk=trk,
                        fishiou=float(fiou[int(ri), int(ci)]),
                        stage_allows_update=bool(cfg.stage2_update_emb),
                        crowd_block=False,
                    )
                    trk.update(det, cfg, update_emb=bool(update_emb))
                    matched_d.add(int(ri))
                    matched_t.add(int(ci))

                # keep only unmatched tracks for later stages
                if matched_t:
                    keep_idx = [i for i in range(len(rem_confirmed_trks2)) if i not in matched_t]
                    rem_confirmed_trks2 = [rem_confirmed_trks2[i] for i in keep_idx]
                    rem_confirmed_idx2 = [rem_confirmed_idx2[i] for i in keep_idx]
                    rem_pred2 = rem_pred2[keep_idx] if keep_idx else np.zeros((0, 4), dtype=np.float32)

        # Stage-3: last chance for remaining high-conf detections
        last_boxes = []
        for t in rem_confirmed_trks2:
            last_boxes.append(t.last_observation.copy() if t.last_observation is not None else t.tlwh.copy())
        last_boxes = np.stack(last_boxes, axis=0).astype(np.float32) if rem_confirmed_trks2 else np.zeros((0, 4), dtype=np.float32)

        m3, ud3, ut3, fiou3 = self._hungarian(
            dets_high_rem2, rem_confirmed_trks2,
            trk_boxes_tlwh=last_boxes,
            app_weight=float(getattr(cfg, "w_app_stage3", 0.0)),
            fishiou_gate=cfg.fishiou_th,
        )
        crowd3: Optional[np.ndarray] = None
        if bool(getattr(cfg, "freeze_emb_in_crowd", False)) and fiou3.size > 0:
            crowd3 = self._crowd_mask_from_fishiou(fiou3)
        for di, tj in m3:
            stage3_allow = bool(getattr(cfg, "stage3_update_emb", True))
            update_emb = self._decide_feature_update(
                stage="stage3_last_observation",
                det=dets_high_rem2[di],
                trk=rem_confirmed_trks2[tj],
                fishiou=float(fiou3[di, tj]),
                stage_allows_update=stage3_allow,
                crowd_block=bool(crowd3 is not None and bool(crowd3[di])),
            )
            rem_confirmed_trks2[tj].update(dets_high_rem2[di], cfg, update_emb=bool(update_emb))

        rem_high2 = [dets_high_rem2[i] for i in ud3]
        rem_confirmed_idx3 = [rem_confirmed_idx2[j] for j in ut3]
        rem_confirmed_trks3 = [trks[j] for j in rem_confirmed_idx3]
        last_boxes3 = []
        for t in rem_confirmed_trks3:
            last_boxes3.append(t.last_observation.copy() if t.last_observation is not None else t.tlwh.copy())
        last_boxes3 = np.stack(last_boxes3, axis=0).astype(np.float32) if rem_confirmed_trks3 else np.zeros((0, 4), dtype=np.float32)

        mlt, ud_lt, ut_lt, fiou_lt = self._long_term_correction(
            rem_high2,
            rem_confirmed_trks3,
            trk_boxes_tlwh=last_boxes3,
            fishiou_gate=float(getattr(cfg, "reid_long_fishiou_gate", 0.05)),
        )
        crowd_lt: Optional[np.ndarray] = None
        if bool(getattr(cfg, "freeze_emb_in_crowd", False)) and fiou_lt.size > 0:
            crowd_lt = self._crowd_mask_from_fishiou(fiou_lt)
        for di, tj in mlt:
            lt_allow = bool(getattr(cfg, "lt_update_emb", True))
            update_emb = self._decide_feature_update(
                stage="reactivation",
                det=rem_high2[di],
                trk=rem_confirmed_trks3[tj],
                fishiou=float(fiou_lt[di, tj]) if fiou_lt.size > 0 else 0.0,
                stage_allows_update=lt_allow,
                crowd_block=bool(crowd_lt is not None and bool(crowd_lt[di])),
                apply_quality_gate=False,
            )
            rem_confirmed_trks3[tj].update(rem_high2[di], cfg, update_emb=bool(update_emb))

        new_dets = [rem_high2[i] for i in ud_lt]
        for d in new_dets:
            t = Track(track_id=self._next_id, tlwh=d.tlwh.copy(), score=float(d.score), emb=None)
            t.last_observation = d.tlwh.copy()
            if d.emb is not None:
                emb = d.emb.astype(np.float32)
                t.emb = emb
                bank_size = int(getattr(cfg, "emb_bank_size", 1))
                if bank_size > 0:
                    t.emb_bank = deque([l2_normalize(emb.reshape(1, -1))[0]], maxlen=bank_size)
                    self._diag_inc("actual_history_writes")
                    self._diag_inc("new_track_history_inits")
            if d.axis is not None:
                ax = d.axis.astype(np.float32)
                ax = l2_normalize(ax.reshape(1, -1))[0]
                t.axis = ax
                bank_size = int(getattr(cfg, "emb_bank_size", 1))
                if bank_size > 0:
                    t.axis_bank = deque([ax], maxlen=bank_size)
            self._record_event(
                stage="new_track",
                track_id=int(t.track_id),
                det=d,
                update_emb=bool(d.emb is not None),
                reason="new_track_init",
                fishiou=None,
                raw_sim=None,
            )
            self._next_id += 1
            self.tracks.append(t)

        self._prune()

        act: List[Track] = []
        score_gate = getattr(cfg, "min_hits_score_gate", None)
        for t in self.tracks:
            if t.time_since_update != 0:
                continue
            if t.hits >= cfg.min_hits:
                act.append(t)
                continue
            if score_gate is not None and float(t.score) >= float(score_gate):
                act.append(t)
        return act

    def _update_single_stage(
        self,
        dets_high: List[Detection],
        dets_low: List[Detection],
        trks: List[Track],
        pred_boxes: np.ndarray,
    ) -> List[Track]:
        """
        Alternative association path used only when cascade is disabled.
        the AMT machinery as close as possible. All detections above det_low_th
        are matched in one Hungarian stage; new tracks are still started only
        from high-confidence unmatched detections.
        """
        cfg = self.cfg
        dets_all = list(dets_high) + list(dets_low)
        m1, ud1, ut1, fiou1 = self._hungarian(
            dets_all,
            trks,
            trk_boxes_tlwh=pred_boxes,
            app_weight=cfg.w_app,
            fishiou_gate=cfg.fishiou_th,
        )
        crowd1: Optional[np.ndarray] = None
        if bool(getattr(cfg, "freeze_emb_in_crowd", False)) and fiou1.size > 0:
            crowd1 = self._crowd_mask_from_fishiou(fiou1)
        for di, tj in m1:
            update_emb = self._decide_feature_update(
                stage="single_stage",
                det=dets_all[di],
                trk=trks[tj],
                fishiou=float(fiou1[di, tj]),
                stage_allows_update=True,
                crowd_block=bool(crowd1 is not None and bool(crowd1[di])),
            )
            trks[tj].update(dets_all[di], cfg, update_emb=bool(update_emb))

        rem_dets = [dets_all[i] for i in ud1]
        rem_high = [d for d in rem_dets if float(d.score) >= float(cfg.det_high_th)]
        rem_trks = [trks[j] for j in ut1]
        last_boxes = []
        for t in rem_trks:
            last_boxes.append(t.last_observation.copy() if t.last_observation is not None else t.tlwh.copy())
        last_boxes = np.stack(last_boxes, axis=0).astype(np.float32) if rem_trks else np.zeros((0, 4), dtype=np.float32)

        mlt, ud_lt, _, fiou_lt = self._long_term_correction(
            rem_high,
            rem_trks,
            trk_boxes_tlwh=last_boxes,
            fishiou_gate=float(getattr(cfg, "reid_long_fishiou_gate", 0.05)),
        )
        crowd_lt: Optional[np.ndarray] = None
        if bool(getattr(cfg, "freeze_emb_in_crowd", False)) and fiou_lt.size > 0:
            crowd_lt = self._crowd_mask_from_fishiou(fiou_lt)
        for di, tj in mlt:
            lt_allow = bool(getattr(cfg, "lt_update_emb", True))
            update_emb = self._decide_feature_update(
                stage="reactivation",
                det=rem_high[di],
                trk=rem_trks[tj],
                fishiou=float(fiou_lt[di, tj]) if fiou_lt.size > 0 else 0.0,
                stage_allows_update=lt_allow,
                crowd_block=bool(crowd_lt is not None and bool(crowd_lt[di])),
                apply_quality_gate=False,
            )
            rem_trks[tj].update(rem_high[di], cfg, update_emb=bool(update_emb))

        new_dets = [rem_high[i] for i in ud_lt]
        for d in new_dets:
            t = Track(track_id=self._next_id, tlwh=d.tlwh.copy(), score=float(d.score), emb=None)
            t.last_observation = d.tlwh.copy()
            if d.emb is not None:
                emb = d.emb.astype(np.float32)
                t.emb = emb
                bank_size = int(getattr(cfg, "emb_bank_size", 1))
                if bank_size > 0:
                    t.emb_bank = deque([l2_normalize(emb.reshape(1, -1))[0]], maxlen=bank_size)
                    self._diag_inc("actual_history_writes")
                    self._diag_inc("new_track_history_inits")
            if d.axis is not None:
                ax = d.axis.astype(np.float32)
                ax = l2_normalize(ax.reshape(1, -1))[0]
                t.axis = ax
                bank_size = int(getattr(cfg, "emb_bank_size", 1))
                if bank_size > 0:
                    t.axis_bank = deque([ax], maxlen=bank_size)
            self._record_event(
                stage="new_track",
                track_id=int(t.track_id),
                det=d,
                update_emb=bool(d.emb is not None),
                reason="new_track_init",
                fishiou=None,
                raw_sim=None,
            )
            self._next_id += 1
            self.tracks.append(t)

        self._prune()

        act: List[Track] = []
        score_gate = getattr(cfg, "min_hits_score_gate", None)
        for t in self.tracks:
            if t.time_since_update != 0:
                continue
            if t.hits >= cfg.min_hits:
                act.append(t)
                continue
            if score_gate is not None and float(t.score) >= float(score_gate):
                act.append(t)
        return act

    def update(self, dets: List[Detection], frame_id: Optional[int] = None) -> List[Track]:
        """
        Update per frame. Returns "active" tracks (time_since_update==0).
        """
        cfg = self.cfg
        if frame_id is None:
            self._frame_id += 1
        else:
            self._frame_id = int(frame_id)
        nms_iou = getattr(cfg, "det_nms_iou", None)
        if nms_iou is not None:
            dets = self._nms_dets(dets, float(nms_iou))

        # prune dead tracks first
        self._prune()

        # active list snapshot
        trks = list(self.tracks)

        # predict once per track
        pred_boxes = []
        for t in trks:
            pred_boxes.append(t.predict(cfg))
        pred_boxes = np.stack(pred_boxes, axis=0).astype(np.float32) if trks else np.zeros((0, 4), dtype=np.float32)

        # split detections
        dets_high = [d for d in dets if d.score >= cfg.det_high_th]
        dets_low = [d for d in dets if cfg.det_low_th <= d.score < cfg.det_high_th]

        if str(getattr(cfg, "association_mode", "cascade")).lower() == "single_stage":
            return self._update_single_stage(dets_high, dets_low, trks, pred_boxes)

        if bool(getattr(cfg, "use_confirmed_cascade", False)):
            return self._update_confirmed_cascade(dets_high, dets_low, trks, pred_boxes)

        # Stage-1: high-conf vs predicted
        m1, ud1, ut1, fiou1 = self._hungarian(
            dets_high, trks,
            trk_boxes_tlwh=pred_boxes,
            app_weight=cfg.w_app,
            fishiou_gate=cfg.fishiou_th,
        )
        crowd1: Optional[np.ndarray] = None
        if bool(getattr(cfg, "freeze_emb_in_crowd", False)) and fiou1.size > 0:
            crowd1 = self._crowd_mask_from_fishiou(fiou1)
        updated_ids = set()
        for di, tj in m1:
            update_emb = self._should_update_emb(dets_high[di], trks[tj], fishiou=float(fiou1[di, tj]))
            if crowd1 is not None and bool(crowd1[di]):
                update_emb = False
            trks[tj].update(dets_high[di], cfg, update_emb=bool(update_emb))
            updated_ids.add(trks[tj].track_id)

        # remaining tracks after stage-1
        rem_trk_idx1 = ut1
        rem_trks1 = [trks[j] for j in rem_trk_idx1]
        rem_pred1 = pred_boxes[rem_trk_idx1] if len(rem_trk_idx1) else np.zeros((0, 4), dtype=np.float32)

        # Stage-2: low-conf vs remaining predicted
        stage2_gate: Union[float, np.ndarray] = float(cfg.fishiou_th)
        if getattr(cfg, "fishiou_th_low_min", None) is not None and getattr(cfg, "fishiou_th_low_max", None) is not None:
            g_min = float(cfg.fishiou_th_low_min)
            g_max = float(cfg.fishiou_th_low_max)
            if g_min > g_max:
                g_min, g_max = g_max, g_min
            s_lo = float(cfg.det_low_th)
            s_hi = float(cfg.det_high_th)
            den = max(1e-6, s_hi - s_lo)
            scores = np.array([float(d.score) for d in dets_low], dtype=np.float32)
            t = np.clip((scores - s_lo) / den, 0.0, 1.0)
            stage2_gate = (g_max - t * (g_max - g_min)).astype(np.float32)
        elif getattr(cfg, "fishiou_th_low", None) is not None:
            stage2_gate = float(cfg.fishiou_th_low)

        if bool(getattr(cfg, "stage2_use_app", True)):
            stage2_app: Union[float, np.ndarray] = float(cfg.w_app_low)
            if getattr(cfg, "w_app_low_min", None) is not None and getattr(cfg, "w_app_low_max", None) is not None:
                w_min = float(cfg.w_app_low_min)
                w_max = float(cfg.w_app_low_max)
                if w_min > w_max:
                    w_min, w_max = w_max, w_min
                s_lo = float(cfg.det_low_th)
                s_hi = float(cfg.det_high_th)
                den = max(1e-6, s_hi - s_lo)
                scores = np.array([float(d.score) for d in dets_low], dtype=np.float32)
                t = np.clip((scores - s_lo) / den, 0.0, 1.0)
                stage2_app = (w_min + t * (w_max - w_min)).astype(np.float32)
        else:
            stage2_app = 0.0
        m2, ud2, ut2, fiou2 = self._hungarian(
            dets_low, rem_trks1,
            trk_boxes_tlwh=rem_pred1,
            app_weight=stage2_app,
            fishiou_gate=stage2_gate,
        )
        crowd2: Optional[np.ndarray] = None
        if bool(getattr(cfg, "freeze_emb_in_crowd", False)) and fiou2.size > 0:
            crowd2 = self._crowd_mask_from_fishiou(fiou2)
        for di, tj in m2:
            update_emb = bool(cfg.stage2_update_emb) and self._should_update_emb(
                dets_low[di], rem_trks1[tj], fishiou=float(fiou2[di, tj])
            )
            if crowd2 is not None and bool(crowd2[di]):
                update_emb = False
            rem_trks1[tj].update(dets_low[di], cfg, update_emb=bool(update_emb))
            updated_ids.add(rem_trks1[tj].track_id)

        # remaining tracks after stage-2
        rem_trk_idx2 = [rem_trk_idx1[j] for j in ut2]  # indices into trks
        rem_trks2 = [trks[j] for j in rem_trk_idx2]
        # remaining high dets after stage-1
        rem_high = [dets_high[i] for i in ud1]

        # Stage-3: last chance (use last_observation boxes)
        last_boxes = []
        for t in rem_trks2:
            last_boxes.append(t.last_observation.copy() if t.last_observation is not None else t.tlwh.copy())
        last_boxes = np.stack(last_boxes, axis=0).astype(np.float32) if rem_trks2 else np.zeros((0, 4), dtype=np.float32)

        m3, ud3, ut3, fiou3 = self._hungarian(
            rem_high, rem_trks2,
            trk_boxes_tlwh=last_boxes,
            app_weight=float(getattr(cfg, "w_app_stage3", 0.0)),
            fishiou_gate=cfg.fishiou_th,
        )
        crowd3: Optional[np.ndarray] = None
        if bool(getattr(cfg, "freeze_emb_in_crowd", False)) and fiou3.size > 0:
            crowd3 = self._crowd_mask_from_fishiou(fiou3)
        for di, tj in m3:
            stage3_allow = bool(getattr(cfg, "stage3_update_emb", True))
            update_emb = stage3_allow and self._should_update_emb(rem_high[di], rem_trks2[tj], fishiou=float(fiou3[di, tj]))
            if crowd3 is not None and bool(crowd3[di]):
                update_emb = False
            rem_trks2[tj].update(rem_high[di], cfg, update_emb=bool(update_emb))
            updated_ids.add(rem_trks2[tj].track_id)

        # remaining for long-term correction
        rem_high2 = [rem_high[i] for i in ud3]
        rem_trks3 = [rem_trks2[j] for j in ut3]
        last_boxes3 = []
        for t in rem_trks3:
            last_boxes3.append(t.last_observation.copy() if t.last_observation is not None else t.tlwh.copy())
        last_boxes3 = np.stack(last_boxes3, axis=0).astype(np.float32) if rem_trks3 else np.zeros((0, 4), dtype=np.float32)

        mlt, ud_lt, ut_lt, fiou_lt = self._long_term_correction(
            rem_high2,
            rem_trks3,
            trk_boxes_tlwh=last_boxes3,
            fishiou_gate=float(getattr(cfg, "reid_long_fishiou_gate", 0.05)),
        )
        crowd_lt: Optional[np.ndarray] = None
        if bool(getattr(cfg, "freeze_emb_in_crowd", False)) and fiou_lt.size > 0:
            crowd_lt = self._crowd_mask_from_fishiou(fiou_lt)
        for di, tj in mlt:
            lt_allow = bool(getattr(cfg, "lt_update_emb", True))
            update_emb = bool(lt_allow)
            if crowd_lt is not None and bool(crowd_lt[di]):
                update_emb = False
            rem_trks3[tj].update(rem_high2[di], cfg, update_emb=bool(update_emb))
            updated_ids.add(rem_trks3[tj].track_id)

        # new tracks from remaining high-conf detections
        new_dets = [rem_high2[i] for i in ud_lt]
        for d in new_dets:
            t = Track(track_id=self._next_id, tlwh=d.tlwh.copy(), score=float(d.score), emb=None)
            t.last_observation = d.tlwh.copy()
            if d.emb is not None:
                emb = d.emb.astype(np.float32)
                t.emb = emb
                bank_size = int(getattr(cfg, "emb_bank_size", 1))
                if bank_size > 0:
                    t.emb_bank = deque([l2_normalize(emb.reshape(1, -1))[0]], maxlen=bank_size)
            if d.axis is not None:
                ax = d.axis.astype(np.float32)
                ax = l2_normalize(ax.reshape(1, -1))[0]
                t.axis = ax
                bank_size = int(getattr(cfg, "emb_bank_size", 1))
                if bank_size > 0:
                    t.axis_bank = deque([ax], maxlen=bank_size)
            self._next_id += 1
            self.tracks.append(t)
            updated_ids.add(t.track_id)

        # prune again
        self._prune()

        # return active tracks (updated in this frame)
        act: List[Track] = []
        score_gate = getattr(cfg, "min_hits_score_gate", None)
        for t in self.tracks:
            if t.time_since_update != 0:
                continue
            if t.hits >= cfg.min_hits:
                act.append(t)
                continue
            if score_gate is not None and float(t.score) >= float(score_gate):
                act.append(t)
        return act
