"""Tracker variants used only for controlled paper comparisons.

The default AMT executable never imports this module. Each option changes one
declared experimental factor while leaving the frozen main tracker untouched.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Optional

import numpy as np

from fishmambatrack.tracking.amt_tracker import (
    AMTTracker,
    AMTTrackerConfig,
    Detection,
    Track,
    iou_matrix,
)


class ControlledAMTTracker(AMTTracker):
    def __init__(self, cfg: AMTTrackerConfig, controls: Mapping[str, Any]):
        self.controls = dict(controls)
        association = dict(self.controls.get("association", {}))
        unknown_association = sorted(
            set(association)
            - {
                "appearance",
                "cascade",
                "crowd_appearance_suppression",
                "geometry",
                "geometry_confidence_scaling",
            }
        )
        if unknown_association:
            raise ValueError(
                f"Unknown controlled association keys: {unknown_association}"
            )
        if not bool(self.controls.get("reactivation", True)):
            cfg = replace(cfg, reid_long_th=2.0)
        super().__init__(cfg)
        self.geometry_name = str(association.get("geometry", "fishiou_plus"))
        if self.geometry_name not in {"fishiou_plus", "iou"}:
            raise ValueError(f"Unsupported controlled geometry: {self.geometry_name}")
        self.use_appearance = bool(association.get("appearance", True))
        self.use_cascade = bool(association.get("cascade", True))
        self.use_crowd_appearance_suppression = bool(
            association.get("crowd_appearance_suppression", True)
        )
        self.use_geometry_confidence = bool(
            association.get("geometry_confidence_scaling", True)
        )
        self.write_policy = str(self.controls.get("write_policy", "reliability_first"))
        valid_write_policies = {
            "all_matches",
            "disabled",
            "reliability_first",
            "reliability_without_crowd_freeze",
            "reliable_single_frame_replacement",
            "stage_eligible",
            "stage_eligible_without_geometry_or_appearance_gates",
            "stage_geometry_appearance",
        }
        if self.write_policy not in valid_write_policies:
            raise ValueError(f"Unsupported write policy: {self.write_policy}")

    def _geometry(self, detections: np.ndarray, tracks: np.ndarray) -> np.ndarray:
        if self.geometry_name == "iou":
            return iou_matrix(detections, tracks)
        return super()._geometry(detections, tracks)

    def _similarity(self, detections, tracks):
        if not self.use_appearance:
            return None
        return super()._similarity(detections, tracks)

    def _raw_similarity(self, detection: Detection, track: Track) -> Optional[float]:
        if not self.use_appearance:
            return None
        return super()._raw_similarity(detection, track)

    def _appearance_weights(
        self, matrix: np.ndarray, base: float, stage: str
    ) -> np.ndarray:
        weights = np.full((matrix.shape[0],), float(base), dtype=np.float32)
        if self.use_crowd_appearance_suppression:
            crowded = super()._crowd_mask(matrix)
            weights[crowded] = np.minimum(weights[crowded], self.cfg.w_app_crowd)
        if self.use_geometry_confidence and stage == "high" and matrix.shape[1] > 0:
            ordered = np.sort(matrix, axis=1)
            top = ordered[:, -1]
            second = ordered[:, -2] if matrix.shape[1] > 1 else np.zeros_like(top)
            confident = (top - second) >= self.cfg.geometry_confidence_margin
            weights[confident] *= self.cfg.geometry_confident_app_factor
        return weights

    def _write_decision(
        self,
        *,
        stage: str,
        detection: Detection,
        track: Track,
        geometry: float,
        crowded: bool,
    ) -> bool:
        policy = self.write_policy
        if policy == "disabled":
            self._inc("successful_matches")
            self._inc("stage_update_disabled")
            self.diagnostic_events.append(
                {
                    "frame": self._frame_id,
                    "stage": stage,
                    "track_id": track.track_id,
                    "score": detection.score,
                    "tlwh": detection.tlwh.astype(float).tolist(),
                    "update_emb": False,
                    "reason": "appearance_disabled",
                    "fishiou": float(geometry),
                    "raw_sim": None,
                }
            )
            return False
        if policy in {"reliability_first", "reliable_single_frame_replacement"}:
            return super()._write_decision(
                stage=stage,
                detection=detection,
                track=track,
                geometry=geometry,
                crowded=crowded,
            )

        self._inc("successful_matches")
        stage_eligible = stage in {"stage1", "stage1_unconfirmed", "reactivation"}
        raw = self._raw_similarity(detection, track)
        reason = "accepted"
        if policy == "all_matches":
            write = True
        else:
            write = stage_eligible
            if not write:
                reason = "stage"
                self._inc("stage_update_disabled")
            else:
                self._inc("candidate_feature_update_events")
        if write and policy in {
            "stage_geometry_appearance",
            "reliability_without_crowd_freeze",
        }:
            if geometry < self.cfg.emb_update_fishiou_th:
                write, reason = False, "geometry"
                self._inc("geometry_gate_rejects")
            elif raw is None or raw < self.cfg.emb_update_sim_th:
                write, reason = False, "appearance"
                self._inc("appearance_gate_rejects")
        if (
            write
            and policy == "stage_eligible_without_geometry_or_appearance_gates"
            and crowded
        ):
            write, reason = False, "crowd"
            self._inc("crowd_gate_rejects")
        if write:
            self._inc("feature_update_gate_pass")
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

    def update(
        self, detections: list[Detection], frame_id: Optional[int] = None
    ) -> list[Track]:
        if self.use_cascade:
            return super().update(detections, frame_id=frame_id)
        self._frame_id = self._frame_id + 1 if frame_id is None else int(frame_id)
        detections = self._nms(
            [det for det in detections if det.score >= self.cfg.det_low_th]
        )
        predicted = (
            np.stack([track.predict(self.cfg) for track in self.tracks])
            if self.tracks
            else np.zeros((0, 4), np.float32)
        )
        matches, unmatched_detections, unmatched_tracks, geometry = self._match(
            detections,
            self.tracks,
            predicted,
            app_weight=self.cfg.w_app,
            gate=self.cfg.fishiou_th,
            stage="high",
        )
        self._apply_matches(matches, detections, self.tracks, geometry, stage="stage1")

        remaining_detections = [detections[index] for index in unmatched_detections]
        remaining_tracks = [self.tracks[index] for index in unmatched_tracks]
        remaining_boxes = (
            predicted[unmatched_tracks]
            if unmatched_tracks
            else np.zeros((0, 4), np.float32)
        )
        similarity = self._similarity(remaining_detections, remaining_tracks)
        reid_geometry = self._geometry(
            (
                np.stack([detection.tlwh for detection in remaining_detections])
                if remaining_detections
                else np.zeros((0, 4), np.float32)
            ),
            remaining_boxes,
        )
        reid_matches: list[tuple[int, int]] = []
        if similarity is not None:
            valid = (similarity >= self.cfg.reid_long_th) & (
                reid_geometry >= self.cfg.reid_long_fishiou_gate
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
            remaining_detections,
            remaining_tracks,
            reid_geometry,
            stage="reactivation",
        )
        matched_reid = {index for index, _ in reid_matches}
        for index, detection in enumerate(remaining_detections):
            if index not in matched_reid and detection.score >= self.cfg.det_high_th:
                self._new_track(detection)
        self.tracks = [
            track
            for track in self.tracks
            if track.time_since_update <= self.cfg.max_age
            and not (track.hits < self.cfg.min_hits and track.time_since_update > 0)
        ]
        return [
            track
            for track in self.tracks
            if track.time_since_update == 0
            and (
                track.hits >= self.cfg.min_hits
                or track.score >= self.cfg.min_hits_score_gate
            )
        ]
