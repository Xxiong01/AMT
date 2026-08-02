"""
fishmambatrack.tracking.kalman_filter

A simple constant-velocity Kalman filter for bbox tracking.

State: [cx, cy, w, h, vx, vy, vw, vh] (8D)
Measurement: [cx, cy, w, h] (4D)

This is enough to run SORT-like motion prediction + gating.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np


class KalmanFilter:
    def __init__(self, dt: float = 1.0) -> None:
        self.ndim = 4
        self.dt = float(dt)

        dim_x = 2 * self.ndim
        dim_z = self.ndim

        # State transition matrix
        self.motion_mat = np.eye(dim_x, dtype=np.float32)
        for i in range(self.ndim):
            self.motion_mat[i, self.ndim + i] = self.dt

        # Measurement matrix
        self.update_mat = np.eye(dim_z, dim_x, dtype=np.float32)

        # Noise weights (relative to object scale)
        self.std_weight_position = 1.0 / 20.0
        self.std_weight_velocity = 1.0 / 160.0

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create track from first measurement.
        measurement: (4,) [cx,cy,w,h]
        """
        mean = np.zeros((8,), dtype=np.float32)
        mean[:4] = measurement.astype(np.float32)
        mean[4:] = 0.0

        s = max(float(measurement[2]), float(measurement[3]), 1.0)

        std_pos = self.std_weight_position * s
        std_vel = self.std_weight_velocity * s

        std = np.array(
            [2 * std_pos, 2 * std_pos, 2 * std_pos, 2 * std_pos,
             10 * std_vel, 10 * std_vel, 10 * std_vel, 10 * std_vel],
            dtype=np.float32
        )
        cov = np.diag(std * std).astype(np.float32)
        return mean, cov

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict next state.
        """
        s = max(float(mean[2]), float(mean[3]), 1.0)
        std_pos = self.std_weight_position * s
        std_vel = self.std_weight_velocity * s

        motion_std = np.array(
            [std_pos, std_pos, std_pos, std_pos,
             std_vel, std_vel, std_vel, std_vel],
            dtype=np.float32
        )
        motion_cov = np.diag(motion_std * motion_std).astype(np.float32)

        mean = self.motion_mat @ mean
        cov = self.motion_mat @ cov @ self.motion_mat.T + motion_cov
        return mean, cov

    def project(self, mean: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project state distribution to measurement space.
        """
        s = max(float(mean[2]), float(mean[3]), 1.0)
        std = self.std_weight_position * s
        R = np.diag((np.array([std, std, std, std], dtype=np.float32) ** 2)).astype(np.float32)

        mean_z = self.update_mat @ mean
        cov_z = self.update_mat @ cov @ self.update_mat.T + R
        return mean_z, cov_z

    def update(self, mean: np.ndarray, cov: np.ndarray, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Correct state with measurement.
        """
        mean_z, cov_z = self.project(mean, cov)

        # Kalman gain
        K = cov @ self.update_mat.T @ np.linalg.inv(cov_z).astype(np.float32)

        innovation = (measurement - mean_z).astype(np.float32)
        new_mean = mean + K @ innovation
        new_cov = cov - K @ cov_z @ K.T
        return new_mean.astype(np.float32), new_cov.astype(np.float32)

    def gating_distance(self, mean: np.ndarray, cov: np.ndarray, measurements: np.ndarray) -> np.ndarray:
        """
        Squared Mahalanobis distance for gating.
        measurements: (N,4)
        return: (N,)
        """
        mean_z, cov_z = self.project(mean, cov)
        chol = np.linalg.cholesky(cov_z).astype(np.float32)

        d = (measurements - mean_z[None, :]).astype(np.float32)  # (N,4)
        z = np.linalg.solve(chol, d.T).astype(np.float32)        # (4,N)
        return np.sum(z * z, axis=0)
