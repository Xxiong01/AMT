"""Controlled tracker variants for the paper's declared ablations.

The production tracker remains unchanged.  This adapter maps each experimental
factor to the complete AMT tracker configuration and overrides only the FIFO
write decision when a write-policy ablation requires it.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from fishmambatrack.tracking.amt_tracker import (
    AMTTracker,
    AMTTrackerConfig,
    Detection,
    FishIoUParams,
    Track,
)


class ControlledAMTTracker(AMTTracker):
    def __init__(self, cfg: AMTTrackerConfig, controls: Mapping[str, Any]):
        self.controls = dict(controls)
        association = dict(self.controls.get("association", {}))
        allowed = {
            "appearance",
            "cascade",
            "crowd_appearance_suppression",
            "geometry",
            "geometry_confidence_scaling",
        }
        unknown = sorted(set(association) - allowed)
        if unknown:
            raise ValueError(f"Unknown controlled association keys: {unknown}")

        geometry = str(association.get("geometry", "fishiou_plus"))
        if geometry not in {"fishiou_plus", "iou"}:
            raise ValueError(f"Unsupported controlled geometry: {geometry}")
        fish = cfg.fishiou_params or FishIoUParams()
        if geometry == "iou":
            fish = replace(fish, mode="iou", adaptive_central=False)
        cfg = replace(cfg, fishiou_params=fish)

        if not bool(association.get("appearance", True)):
            cfg = replace(
                cfg,
                use_reid=False,
                w_app=0.0,
                w_app_low=0.0,
                w_app_stage3=0.0,
                w_app_crowd=None,
                freeze_emb_in_crowd=False,
                reid_long_th=2.0,
            )
        if not bool(association.get("cascade", True)):
            cfg = replace(cfg, association_mode="single_stage")
        if not bool(association.get("crowd_appearance_suppression", True)):
            # This leave-one-out variant removes crowd-conditioned appearance
            # damping but intentionally retains crowd-based FIFO freezing.
            cfg = replace(cfg, w_app_crowd=None)
        if not bool(association.get("geometry_confidence_scaling", True)):
            cfg = replace(cfg, geometry_confident_app_scale=False)
        if not bool(self.controls.get("reactivation", True)):
            cfg = replace(cfg, reid_long_th=2.0)

        self.write_policy = str(
            self.controls.get("write_policy", "reliability_first")
        )
        valid = {
            "all_matches",
            "disabled",
            "reliability_first",
            "reliability_without_crowd_freeze",
            "reliable_single_frame_replacement",
            "stage_eligible",
            "stage_eligible_without_geometry_or_appearance_gates",
            "stage_geometry_appearance",
        }
        if self.write_policy not in valid:
            raise ValueError(f"Unsupported write policy: {self.write_policy}")
        super().__init__(cfg)

    @staticmethod
    def _stage_is_eligible(stage: str) -> bool:
        return str(stage).startswith("stage1") or str(stage) == "reactivation"

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
        policy = self.write_policy
        if policy in {"reliability_first", "reliable_single_frame_replacement"}:
            return super()._decide_feature_update(
                stage=stage,
                det=det,
                trk=trk,
                fishiou=fishiou,
                stage_allows_update=stage_allows_update,
                crowd_block=crowd_block,
                apply_quality_gate=apply_quality_gate,
            )
        if policy == "disabled":
            return super()._decide_feature_update(
                stage=stage,
                det=det,
                trk=trk,
                fishiou=fishiou,
                stage_allows_update=False,
                crowd_block=False,
                apply_quality_gate=False,
            )
        if policy == "all_matches":
            return super()._decide_feature_update(
                stage=stage,
                det=det,
                trk=trk,
                fishiou=fishiou,
                stage_allows_update=True,
                crowd_block=False,
                apply_quality_gate=False,
            )

        eligible = self._stage_is_eligible(stage)
        if policy == "stage_eligible":
            quality_gate = False
            block_crowd = False
        elif policy == "stage_geometry_appearance":
            quality_gate = True
            block_crowd = False
        elif policy == "reliability_without_crowd_freeze":
            quality_gate = True
            block_crowd = False
        elif policy == "stage_eligible_without_geometry_or_appearance_gates":
            quality_gate = False
            block_crowd = bool(crowd_block)
        else:  # guarded by the valid-policy check above
            raise AssertionError(policy)

        return super()._decide_feature_update(
            stage=stage,
            det=det,
            trk=trk,
            fishiou=fishiou,
            stage_allows_update=eligible,
            crowd_block=block_crowd,
            apply_quality_gate=quality_gate,
        )
