"""
fishmambatrack.tracking.tracklet

Tracklet container with feature bank + selective feature update.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np

from .kalman_filter import KalmanFilter


def tlwh_to_xywh(tlwh: Tuple[float, float, float, float]) -> np.ndarray:
    x, y, w, h = tlwh
    cx = x + 0.5 * w
    cy = y + 0.5 * h
    return np.array([cx, cy, w, h], dtype=np.float32)


def xywh_to_tlwh(xywh: np.ndarray) -> Tuple[float, float, float, float]:
    cx, cy, w, h = xywh.tolist()
    x = cx - 0.5 * w
    y = cy - 0.5 * h
    return (float(x), float(y), float(w), float(h))


def l2norm(x: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(x) + 1e-12)
    return (x / n).astype(np.float32)


@dataclass
class Tracklet:
    track_id: int
    mean: np.ndarray          # (8,)
    cov: np.ndarray           # (8,8)
    feature: Optional[np.ndarray]  # (D,) EMA feature, L2 normalized
    score: float

    age: int = 1
    time_since_update: int = 0
    hits: int = 1
    confirmed: bool = False

    # Feature bank (gallery): store recent features, distance uses min cosine distance
    feature_bank: List[np.ndarray] = field(default_factory=list)

    def predict(self, kf: KalmanFilter) -> None:
        self.mean, self.cov = kf.predict(self.mean, self.cov)
        self.age += 1
        self.time_since_update += 1

    def update(
        self,
        kf: KalmanFilter,
        tlwh: Tuple[float, float, float, float],
        score: float,
        feature: Optional[np.ndarray],
        *,
        ema_alpha: float = 0.9,
        min_hits: int = 1,
        update_feature: bool = True,
        bank_size: int = 30,
    ) -> None:
        z = tlwh_to_xywh(tlwh)
        self.mean, self.cov = kf.update(self.mean, self.cov, z)

        self.score = float(score)
        self.time_since_update = 0
        self.hits += 1
        if (not self.confirmed) and (self.hits >= int(min_hits)):
            self.confirmed = True

        if feature is None:
            return

        f = l2norm(feature.astype(np.float32))

        # Always initialize feature/bank for a newborn track
        if self.feature is None:
            self.feature = f
            self.feature_bank.append(f)
            if len(self.feature_bank) > int(bank_size):
                self.feature_bank = self.feature_bank[-int(bank_size):]
            return

        if update_feature:
            # EMA update
            self.feature = l2norm((ema_alpha * self.feature + (1.0 - ema_alpha) * f).astype(np.float32))
            # Bank update
            self.feature_bank.append(f)
            if len(self.feature_bank) > int(bank_size):
                self.feature_bank = self.feature_bank[-int(bank_size):]

    def to_tlwh(self) -> Tuple[float, float, float, float]:
        xywh = self.mean[:4]
        return xywh_to_tlwh(xywh)
