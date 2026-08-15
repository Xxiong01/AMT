"""Compact AMT association and reliability-first identity-memory updates."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def l2_normalize(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return values / (np.linalg.norm(values, axis=1, keepdims=True) + eps)


def tlwh_to_xyxy(values: np.ndarray) -> np.ndarray:
    result = values.astype(np.float32, copy=True)
    result[:, 2:] += result[:, :2]
    return result


def iou_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if len(left) == 0 or len(right) == 0:
        return np.zeros((len(left), len(right)), dtype=np.float32)
    left, right = tlwh_to_xyxy(left), tlwh_to_xyxy(right)
    x1 = np.maximum(left[:, None, 0], right[None, :, 0])
    y1 = np.maximum(left[:, None, 1], right[None, :, 1])
    x2 = np.minimum(left[:, None, 2], right[None, :, 2])
    y2 = np.minimum(left[:, None, 3], right[None, :, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    left_area = np.maximum(0.0, left[:, 2] - left[:, 0]) * np.maximum(
        0.0, left[:, 3] - left[:, 1]
    )
    right_area = np.maximum(0.0, right[:, 2] - right[:, 0]) * np.maximum(
        0.0, right[:, 3] - right[:, 1]
    )
    return (
        intersection / (left_area[:, None] + right_area[None, :] - intersection + 1e-9)
    ).astype(np.float32)


@dataclass
class FishIoUParams:
    alpha: float = 0.15
    beta: float = 0.30
    gamma: float = 0.25
    ar_ref: float = 2.0
    ar_scale_min: float = 0.6
    ar_scale_max: float = 1.4
    w1: float = 1.0
    w2: float = 0.3
    w3: float = 0.1
    w4: float = 0.2
    w5: float = 0.4


def fishiou_matrix(
    detections: np.ndarray, tracks: np.ndarray, params: FishIoUParams
) -> np.ndarray:
    overlap = iou_matrix(detections, tracks)
    if len(detections) == 0 or len(tracks) == 0:
        return overlap
    dxy, txy = tlwh_to_xyxy(detections), tlwh_to_xyxy(tracks)
    dc = dxy[:, None, :]
    tc = txy[None, :, :]
    det_center = 0.5 * (dxy[:, :2] + dxy[:, 2:])
    trk_center = 0.5 * (txy[:, :2] + txy[:, 2:])
    enclosing_width = np.maximum(dc[:, :, 2], tc[:, :, 2]) - np.minimum(
        dc[:, :, 0], tc[:, :, 0]
    )
    enclosing_height = np.maximum(dc[:, :, 3], tc[:, :, 3]) - np.minimum(
        dc[:, :, 1], tc[:, :, 1]
    )
    center_distance = np.sum((det_center[:, None] - trk_center[None]) ** 2, axis=2)
    center_score = 1.0 - center_distance / (
        enclosing_width**2 + enclosing_height**2 + 1e-9
    )

    def central(boxes: np.ndarray) -> np.ndarray:
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ratio = widths / (heights + 1e-9)
        scale = np.clip(
            params.ar_ref / (ratio + 1e-9),
            params.ar_scale_min,
            params.ar_scale_max,
        )
        result = boxes.copy()
        result[:, 0] += params.alpha * scale * widths
        result[:, 1] += params.beta * heights
        result[:, 2] -= params.gamma * scale * widths
        result[:, 3] -= params.beta * heights
        result[:, 2] = np.maximum(result[:, 2], result[:, 0] + 1e-3)
        result[:, 3] = np.maximum(result[:, 3], result[:, 1] + 1e-3)
        result[:, 2:] -= result[:, :2]
        return result

    central_overlap = iou_matrix(central(dxy), central(txy))
    det_ratio = detections[:, 2] / (detections[:, 3] + 1e-9)
    trk_ratio = tracks[:, 2] / (tracks[:, 3] + 1e-9)
    ratio_score = np.exp(
        -np.abs(np.log((det_ratio[:, None] + 1e-9) / (trk_ratio[None] + 1e-9)))
    )
    det_area = detections[:, 2] * detections[:, 3]
    trk_area = tracks[:, 2] * tracks[:, 3]
    area_score = np.minimum(det_area[:, None], trk_area[None]) / (
        np.maximum(det_area[:, None], trk_area[None]) + 1e-9
    )
    weighted = (
        params.w1 * overlap
        + params.w2 * np.clip(center_score, 0.0, 1.0)
        + params.w3 * central_overlap
        + params.w4 * ratio_score
        + params.w5 * area_score
    ) / max(1e-9, params.w1 + params.w2 + params.w3 + params.w4 + params.w5)
    return np.clip(weighted, 0.0, 1.0).astype(np.float32)


@dataclass
class AMTTrackerConfig:
    det_low_th: float = 0.15
    det_high_th: float = 0.60
    det_nms_iou: float = 0.90
    fishiou_th: float = 0.25
    w_fishiou: float = 1.0
    w_app: float = 1.25
    w_app_low: float = 0.50
    w_app_stage3: float = 0.0
    w_app_crowd: float = 0.45
    crowd_fishiou_th: float = 0.06
    crowd_count_th: int = 2
    geometry_confidence_margin: float = 0.04
    geometry_confident_app_factor: float = 0.50
    emb_update_sim_th: float = 0.475
    emb_update_fishiou_th: float = 0.40
    reid_long_th: float = 0.735
    reid_long_fishiou_gate: float = 0.08
    max_age: int = 30
    min_hits: int = 2
    min_hits_score_gate: float = 0.80
    inertia: float = 0.90
    temporal_memory_length: int = 48
    fishiou_params: Optional[FishIoUParams] = None


@dataclass
class Detection:
    tlwh: np.ndarray
    score: float
    emb: Optional[np.ndarray] = None
    temporal_query_emb: Optional[np.ndarray] = None
    frame_feature: Optional[np.ndarray] = None


@dataclass
class Track:
    track_id: int
    tlwh: np.ndarray
    score: float
    appearance_bank: Optional[np.ndarray] = None
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    last_observation: Optional[np.ndarray] = None
    velocity: Optional[np.ndarray] = None
    temporal_history_depth: int = 0
    temporal_history: Deque[np.ndarray] = field(default_factory=deque)

    def predict(self, cfg: AMTTrackerConfig) -> np.ndarray:
        self.age += 1
        self.time_since_update += 1
        predicted = self.tlwh.copy()
        if self.velocity is not None:
            predicted[:2] += self.velocity
        return predicted

    def update(self, detection: Detection, cfg: AMTTrackerConfig, write: bool) -> None:
        previous_center = self.tlwh[:2] + 0.5 * self.tlwh[2:]
        current_center = detection.tlwh[:2] + 0.5 * detection.tlwh[2:]
        displacement = (current_center - previous_center).astype(np.float32)
        self.velocity = (
            displacement
            if self.velocity is None
            else (cfg.inertia * self.velocity + (1.0 - cfg.inertia) * displacement)
        )
        self.tlwh = detection.tlwh.copy()
        self.last_observation = detection.tlwh.copy()
        self.score = float(detection.score)
        self.hits += 1
        self.time_since_update = 0
        if write and detection.emb is not None:
            self.appearance_bank = l2_normalize(
                np.asarray(detection.emb, dtype=np.float32).reshape(1, -1)
            )[0]
        if write and detection.frame_feature is not None:
            self.temporal_history.append(
                np.asarray(detection.frame_feature, dtype=np.float32).copy()
            )
            self.temporal_history_depth = len(self.temporal_history)


class AMTTracker:
    def __init__(self, cfg: AMTTrackerConfig):
        if cfg.temporal_memory_length <= 0:
            raise ValueError("temporal_memory_length must be positive.")
        if cfg.crowd_count_th < 1:
            raise ValueError("crowd_count_th must be at least 1.")
        self.cfg = cfg
        self.fishiou_params = cfg.fishiou_params or FishIoUParams()
        self.tracks: List[Track] = []
        self._next_id = 1
        self._frame_id = 0
        self.diagnostic_events: List[Dict[str, object]] = []
        self.diagnostics: Dict[str, int] = {}
        self._reset_diagnostics()

    def _reset_diagnostics(self) -> None:
        self.diagnostics = {
            key: 0
            for key in (
                "successful_matches",
                "candidate_feature_update_events",
                "feature_update_gate_pass",
                "stage_update_disabled",
                "geometry_gate_rejects",
                "appearance_gate_rejects",
                "crowd_gate_rejects",
                "actual_history_writes",
                "new_track_history_inits",
                "reactivation_attempts",
                "reactivation_matches",
                "reactivation_history_writes",
            )
        }

    def _inc(self, key: str, value: int = 1) -> None:
        self.diagnostics[key] = self.diagnostics.get(key, 0) + int(value)

    def _similarity(
        self, detections: Sequence[Detection], tracks: Sequence[Track]
    ) -> Optional[np.ndarray]:
        if not detections or not tracks:
            return None
        if any(track.appearance_bank is None for track in tracks) or any(
            det.emb is None for det in detections
        ):
            return None
        detection_values = np.stack(
            [
                (
                    det.temporal_query_emb
                    if det.temporal_query_emb is not None
                    else det.emb
                )
                for det in detections
            ]
        ).astype(np.float32)
        track_values = np.stack([track.appearance_bank for track in tracks]).astype(
            np.float32
        )
        similarity = l2_normalize(detection_values) @ l2_normalize(track_values).T
        return np.clip(similarity, 0.0, 1.0)

    def _raw_similarity(self, detection: Detection, track: Track) -> Optional[float]:
        if detection.emb is None or track.appearance_bank is None:
            return None
        left = l2_normalize(np.asarray(detection.emb).reshape(1, -1))[0]
        right = l2_normalize(np.asarray(track.appearance_bank).reshape(1, -1))[0]
        return float(np.clip(left @ right, 0.0, 1.0))

    def _crowd_mask(self, matrix: np.ndarray) -> np.ndarray:
        if matrix.size == 0:
            return np.zeros((matrix.shape[0],), dtype=bool)
        # The OFAT protocol reserves K_crowd=1 as the explicit crowd-off
        # sentinel.  The frozen main method uses K_crowd=2.
        if self.cfg.crowd_count_th == 1:
            return np.zeros((matrix.shape[0],), dtype=bool)
        return (matrix >= self.cfg.crowd_fishiou_th).sum(
            axis=1
        ) >= self.cfg.crowd_count_th

    def _geometry(self, detections: np.ndarray, tracks: np.ndarray) -> np.ndarray:
        """Return the main method's FishIoU+ association geometry."""
        return fishiou_matrix(detections, tracks, self.fishiou_params)

    def _appearance_weights(
        self, matrix: np.ndarray, base: float, stage: str
    ) -> np.ndarray:
        weights = np.full((matrix.shape[0],), float(base), dtype=np.float32)
        crowded = self._crowd_mask(matrix)
        weights[crowded] = np.minimum(weights[crowded], self.cfg.w_app_crowd)
        if stage == "high" and matrix.shape[1] > 0:
            ordered = np.sort(matrix, axis=1)
            top = ordered[:, -1]
            second = ordered[:, -2] if matrix.shape[1] > 1 else np.zeros_like(top)
            confident = (top - second) >= self.cfg.geometry_confidence_margin
            weights[confident] *= self.cfg.geometry_confident_app_factor
        return weights

    def _match(
        self,
        detections: Sequence[Detection],
        tracks: Sequence[Track],
        boxes: np.ndarray,
        *,
        app_weight: float,
        gate: float,
        stage: str,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int], np.ndarray]:
        if not detections or not tracks:
            return (
                [],
                list(range(len(detections))),
                list(range(len(tracks))),
                np.zeros((len(detections), len(tracks)), np.float32),
            )
        detection_boxes = np.stack([det.tlwh for det in detections]).astype(np.float32)
        geometry = self._geometry(detection_boxes, boxes)
        score = self.cfg.w_fishiou * geometry
        appearance = self._similarity(detections, tracks)
        if appearance is not None and app_weight > 0:
            score += (
                self._appearance_weights(geometry, app_weight, stage)[:, None]
                * appearance
            )
        cost = -score
        cost[geometry < gate] = 1e6
        rows, columns = linear_sum_assignment(cost)
        matches = [
            (int(row), int(column))
            for row, column in zip(rows, columns)
            if cost[row, column] < 1e5
        ]
        matched_d = {row for row, _ in matches}
        matched_t = {column for _, column in matches}
        return (
            matches,
            [index for index in range(len(detections)) if index not in matched_d],
            [index for index in range(len(tracks)) if index not in matched_t],
            geometry,
        )

    def _write_decision(
        self,
        *,
        stage: str,
        detection: Detection,
        track: Track,
        geometry: float,
        crowded: bool,
    ) -> bool:
        self._inc("successful_matches")
        reason = "accepted"
        write = stage in {"stage1", "stage1_unconfirmed", "reactivation"}
        raw = self._raw_similarity(detection, track)
        if not write:
            self._inc("stage_update_disabled")
            reason = "stage"
        else:
            self._inc("candidate_feature_update_events")
            if geometry < self.cfg.emb_update_fishiou_th:
                write, reason = False, "geometry"
                self._inc("geometry_gate_rejects")
            elif raw is None or raw < self.cfg.emb_update_sim_th:
                write, reason = False, "appearance"
                self._inc("appearance_gate_rejects")
            else:
                self._inc("feature_update_gate_pass")
        if write and crowded:
            write, reason = False, "crowd"
            self._inc("crowd_gate_rejects")
        if write:
            self._inc("actual_history_writes")
            if stage == "reactivation":
                self._inc("reactivation_history_writes")
        self.diagnostic_events.append(
            {
                "frame": self._frame_id,
                "stage": stage,
                "track_id": track.track_id,
                "score": detection.score,
                "tlwh": detection.tlwh.astype(float).tolist(),
                "update_emb": write,
                "reason": reason,
                "fishiou": float(geometry),
                "raw_sim": raw,
            }
        )
        return write

    def _apply_matches(
        self,
        matches: Sequence[Tuple[int, int]],
        detections: Sequence[Detection],
        tracks: Sequence[Track],
        geometry: np.ndarray,
        *,
        stage: str,
    ) -> None:
        crowded = self._crowd_mask(geometry)
        for detection_index, track_index in matches:
            write = self._write_decision(
                stage=stage,
                detection=detections[detection_index],
                track=tracks[track_index],
                geometry=float(geometry[detection_index, track_index]),
                crowded=bool(crowded[detection_index]),
            )
            tracks[track_index].update(detections[detection_index], self.cfg, write)

    def _nms(self, detections: Sequence[Detection]) -> List[Detection]:
        threshold = self.cfg.det_nms_iou
        if len(detections) < 2:
            return list(detections)
        order = sorted(
            range(len(detections)),
            key=lambda index: detections[index].score,
            reverse=True,
        )
        kept: List[int] = []
        boxes = np.stack([det.tlwh for det in detections])
        while order:
            index = order.pop(0)
            kept.append(index)
            if order:
                overlap = iou_matrix(boxes[index : index + 1], boxes[order])[0]
                order = [
                    candidate
                    for candidate, value in zip(order, overlap)
                    if value <= threshold
                ]
        return [detections[index] for index in kept]

    def _new_track(self, detection: Detection) -> None:
        embedding = (
            None
            if detection.emb is None
            else l2_normalize(np.asarray(detection.emb).reshape(1, -1))[0]
        )
        track = Track(
            track_id=self._next_id,
            tlwh=detection.tlwh.copy(),
            score=detection.score,
            appearance_bank=embedding,
            last_observation=detection.tlwh.copy(),
            temporal_history=deque(maxlen=self.cfg.temporal_memory_length),
        )
        if detection.frame_feature is not None:
            track.temporal_history.append(
                np.asarray(detection.frame_feature, dtype=np.float32).copy()
            )
            track.temporal_history_depth = 1
        self._next_id += 1
        self.tracks.append(track)
        if embedding is not None:
            self._inc("actual_history_writes")
            self._inc("new_track_history_inits")
        self.diagnostic_events.append(
            {
                "frame": self._frame_id,
                "stage": "new_track",
                "track_id": track.track_id,
                "score": detection.score,
                "tlwh": detection.tlwh.astype(float).tolist(),
                "update_emb": embedding is not None,
                "reason": "initial_observation",
                "fishiou": None,
                "raw_sim": None,
            }
        )

    def update(
        self, detections: List[Detection], frame_id: Optional[int] = None
    ) -> List[Track]:
        self._frame_id = self._frame_id + 1 if frame_id is None else int(frame_id)
        detections = self._nms(
            [det for det in detections if det.score >= self.cfg.det_low_th]
        )
        high = [det for det in detections if det.score >= self.cfg.det_high_th]
        low = [
            det
            for det in detections
            if self.cfg.det_low_th <= det.score < self.cfg.det_high_th
        ]
        predicted = (
            np.stack([track.predict(self.cfg) for track in self.tracks])
            if self.tracks
            else np.zeros((0, 4), np.float32)
        )

        confirmed_indices = [
            index
            for index, track in enumerate(self.tracks)
            if track.hits >= self.cfg.min_hits
        ]
        unconfirmed_indices = [
            index
            for index, track in enumerate(self.tracks)
            if track.hits < self.cfg.min_hits
        ]
        confirmed = [self.tracks[index] for index in confirmed_indices]
        unconfirmed = [self.tracks[index] for index in unconfirmed_indices]
        confirmed_boxes = (
            predicted[confirmed_indices]
            if confirmed_indices
            else np.zeros((0, 4), np.float32)
        )
        unconfirmed_boxes = (
            predicted[unconfirmed_indices]
            if unconfirmed_indices
            else np.zeros((0, 4), np.float32)
        )

        first, remaining_high, remaining_confirmed, geometry1 = self._match(
            high,
            confirmed,
            confirmed_boxes,
            app_weight=self.cfg.w_app,
            gate=self.cfg.fishiou_th,
            stage="high",
        )
        self._apply_matches(first, high, confirmed, geometry1, stage="stage1")

        high_after_first = [high[index] for index in remaining_high]
        second, remaining_high2, _, geometry1b = self._match(
            high_after_first,
            unconfirmed,
            unconfirmed_boxes,
            app_weight=0.0,
            gate=self.cfg.fishiou_th,
            stage="high",
        )
        self._apply_matches(
            second,
            high_after_first,
            unconfirmed,
            geometry1b,
            stage="stage1_unconfirmed",
        )
        unmatched_high = [high_after_first[index] for index in remaining_high2]
        remaining_tracks = [confirmed[index] for index in remaining_confirmed]
        remaining_boxes = (
            confirmed_boxes[remaining_confirmed]
            if remaining_confirmed
            else np.zeros((0, 4), np.float32)
        )

        third, _, remaining_tracks2, geometry2 = self._match(
            low,
            remaining_tracks,
            remaining_boxes,
            app_weight=self.cfg.w_app_low,
            gate=self.cfg.fishiou_th,
            stage="low",
        )
        self._apply_matches(third, low, remaining_tracks, geometry2, stage="stage2")

        stage3_tracks = [remaining_tracks[index] for index in remaining_tracks2]
        last_boxes = (
            np.stack(
                [
                    (
                        track.last_observation
                        if track.last_observation is not None
                        else track.tlwh
                    )
                    for track in stage3_tracks
                ]
            )
            if stage3_tracks
            else np.zeros((0, 4), np.float32)
        )
        fourth, remaining_high3, remaining_tracks3, geometry3 = self._match(
            unmatched_high,
            stage3_tracks,
            last_boxes,
            app_weight=self.cfg.w_app_stage3,
            gate=self.cfg.fishiou_th,
            stage="high",
        )
        self._apply_matches(
            fourth,
            unmatched_high,
            stage3_tracks,
            geometry3,
            stage="stage3",
        )

        reid_detections = [unmatched_high[index] for index in remaining_high3]
        reid_tracks = [stage3_tracks[index] for index in remaining_tracks3]
        reid_boxes = (
            np.stack(
                [
                    (
                        track.last_observation
                        if track.last_observation is not None
                        else track.tlwh
                    )
                    for track in reid_tracks
                ]
            )
            if reid_tracks
            else np.zeros((0, 4), np.float32)
        )
        similarity = self._similarity(reid_detections, reid_tracks)
        geometry_reid = self._geometry(
            (
                np.stack([det.tlwh for det in reid_detections])
                if reid_detections
                else np.zeros((0, 4), np.float32)
            ),
            reid_boxes,
        )
        reid_matches: List[Tuple[int, int]] = []
        if similarity is not None:
            valid = (similarity >= self.cfg.reid_long_th) & (
                geometry_reid >= self.cfg.reid_long_fishiou_gate
            )
            self._inc("reactivation_attempts", int(valid.sum()))
            candidates = [
                (float(similarity[row, column]), int(row), int(column))
                for row, column in np.argwhere(valid)
            ]
            used_detections: set[int] = set()
            used_tracks: set[int] = set()
            for _, row, column in sorted(candidates, reverse=True):
                if row in used_detections or column in used_tracks:
                    continue
                reid_matches.append((row, column))
                used_detections.add(row)
                used_tracks.add(column)
            self._inc("reactivation_matches", len(reid_matches))
        self._apply_matches(
            reid_matches,
            reid_detections,
            reid_tracks,
            geometry_reid,
            stage="reactivation",
        )
        matched_reid = {index for index, _ in reid_matches}
        for index, detection in enumerate(reid_detections):
            if index not in matched_reid:
                self._new_track(detection)

        retained = []
        for track in self.tracks:
            if track.time_since_update <= self.cfg.max_age:
                if track.hits < self.cfg.min_hits and track.time_since_update > 0:
                    continue
                retained.append(track)
        self.tracks = retained
        return [
            track
            for track in self.tracks
            if track.time_since_update == 0
            and (
                track.hits >= self.cfg.min_hits
                or track.score >= self.cfg.min_hits_score_gate
            )
        ]
