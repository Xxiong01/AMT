#!/usr/bin/env python3
"""Validate the public AMT method, checkpoint, and experiment manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _require_equal(
    failures: list[str], label: str, actual: object, expected: object
) -> None:
    if actual != expected:
        failures.append(f"{label}: expected {expected!r}, found {actual!r}")


def _load_yaml(path: Path, failures: list[str]) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        failures.append(f"Invalid YAML {path.relative_to(ROOT)}: {exc}")
        return {}
    if not isinstance(value, dict):
        failures.append(f"YAML root is not a mapping: {path.relative_to(ROOT)}")
        return {}
    return value


def _validate_python_and_yaml(failures: list[str]) -> dict[Path, dict[str, Any]]:
    for path in sorted(ROOT.rglob("*.py")):
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except Exception as exc:
            failures.append(
                f"Python compilation failed for {path.relative_to(ROOT)}: {exc}"
            )
    return {path: _load_yaml(path, failures) for path in sorted(ROOT.rglob("*.yaml"))}


def _validate_main_config(
    failures: list[str], yaml_values: dict[Path, dict[str, Any]]
) -> None:
    tracker_path = ROOT / "configs" / "tracker" / "amt_l48.yaml"
    tracker_doc = yaml_values.get(tracker_path, {})
    expected_tracker = {
        "det_low_th": 0.15,
        "det_high_th": 0.60,
        "det_nms_iou": 0.90,
        "fishiou_th": 0.25,
        "w_fishiou": 1.0,
        "w_app": 1.25,
        "w_app_low": 0.50,
        "w_app_stage3": 0.0,
        "w_app_crowd": 0.45,
        "crowd_fishiou_th": 0.06,
        "crowd_count_th": 2,
        "geometry_confidence_margin": 0.04,
        "geometry_confident_app_factor": 0.50,
        "emb_update_sim_th": 0.475,
        "emb_update_fishiou_th": 0.40,
        "reid_long_th": 0.735,
        "reid_long_fishiou_gate": 0.08,
        "max_age": 30,
        "min_hits": 2,
        "min_hits_score_gate": 0.80,
        "inertia": 0.90,
        "fishiou_params": {
            "alpha": 0.15,
            "beta": 0.30,
            "gamma": 0.25,
            "ar_ref": 2.0,
            "ar_scale_min": 0.6,
            "ar_scale_max": 1.4,
            "w1": 1.0,
            "w2": 0.3,
            "w3": 0.1,
            "w4": 0.2,
            "w5": 0.4,
        },
    }
    _require_equal(
        failures,
        "frozen tracker parameters",
        tracker_doc.get("tracker"),
        expected_tracker,
    )
    _require_equal(
        failures,
        "main temporal memory",
        tracker_doc.get("temporal_memory"),
        {"length": 48},
    )

    model_path = ROOT / "configs" / "models" / "mamba_l48.yaml"
    model_doc = yaml_values.get(model_path, {})
    _require_equal(
        failures, "main model name", model_doc.get("model", {}).get("name"), "mamba"
    )
    _require_equal(
        failures, "main model length", model_doc.get("sequence", {}).get("length"), 48
    )
    _require_equal(
        failures,
        "main pool mode",
        model_doc.get("model", {}).get("pool_mode"),
        "mean_last",
    )

    forbidden_main_keys = {
        "detection_linked_history",
        "offline_iou_history",
        "single_frame_fallback",
        "repeat_current_frame",
        "current_frame_repetition",
        "history_fusion",
        "query_adapter",
        "history_adapter",
        "legacy_tracker",
    }
    serialized = yaml.safe_dump(tracker_doc, sort_keys=True).lower()
    for key in forbidden_main_keys:
        if key in serialized:
            failures.append(f"Experimental control leaked into the main config: {key}")


def _validate_main_semantics(failures: list[str]) -> None:
    online = (ROOT / "fishmambatrack" / "runtime" / "online_amt.py").read_text(
        encoding="utf-8"
    )
    tracker = (ROOT / "fishmambatrack" / "tracking" / "amt_tracker.py").read_text(
        encoding="utf-8"
    )
    cli = (ROOT / "scripts" / "track_amt.py").read_text(encoding="utf-8")

    required_online = (
        "def encode_current_frame_detections(",
        "features.unsqueeze(1)",
        "track.temporal_history",
        "_pad_earliest",
        "changed: List[int]",
    )
    required_tracker = (
        "temporal_history: Deque[np.ndarray]",
        "deque(maxlen=self.cfg.temporal_memory_length)",
        'stage in {"stage1", "stage1_unconfirmed", "reactivation"}',
        'stage="stage2"',
        'stage="stage3"',
        'stage="reactivation"',
        "geometry < self.cfg.emb_update_fishiou_th",
        "raw < self.cfg.emb_update_sim_th",
        "if write and crowded",
    )
    for fragment in required_online:
        if fragment not in online:
            failures.append(f"Missing online-method invariant: {fragment}")
    for fragment in required_tracker:
        if fragment not in tracker:
            failures.append(f"Missing tracker invariant: {fragment}")

    forbidden_main_source = (
        "encode_sequence_detections",
        "detection_linked_history",
        "offline_iou_history",
        "single_frame_fallback",
        "repeat_current_frame",
        "history_fusion",
        "query_adapter",
        "history_adapter",
    )
    main_source = (online + "\n" + cli).lower()
    for fragment in forbidden_main_source:
        if fragment in main_source:
            failures.append(
                f"Controlled comparison leaked into the main entry point: {fragment}"
            )
    if "experiments" in cli.lower():
        failures.append(
            "The main tracking entry point imports the controlled-experiment package."
        )


def _validate_checkpoint(failures: list[str]) -> None:
    checkpoint_path = ROOT / "checkpoints" / "best.pt"
    if not checkpoint_path.is_file():
        failures.append("Missing checkpoints/best.pt")
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_meta = {
        "model_name": "mamba",
        "sequence_length": 48,
        "pool_mode": "mean_last",
        "embedding_dimension": 256,
        "training_seed": 0,
        "selection_split": "MFT25-Train-derived Dev",
        "final_val_used_for_selection": False,
        "input_size": [128, 256],
        "crop_padding": 0.10,
        "model_cfg": {
            "mamba_d_model": 256,
            "mamba_layers": 2,
            "mamba_d_state": 16,
            "mamba_d_conv": 4,
            "mamba_expand": 2,
            "mamba_dropout": 0.10,
            "max_seq_len": 48,
            "emb_dim": 256,
        },
    }
    _require_equal(
        failures, "checkpoint metadata", checkpoint.get("meta"), expected_meta
    )
    if sorted(checkpoint) != ["meta", "model"]:
        failures.append(
            "The release checkpoint must contain only model tensors and normalized metadata."
        )

    try:
        import sys

        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from fishmambatrack.models.reid.registry import load_checkpoint
        import fishmambatrack

        imported_root = Path(fishmambatrack.__file__).resolve().parents[1]
        if imported_root != ROOT:
            failures.append(
                f"Imported fishmambatrack from {imported_root}, expected {ROOT}."
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _ = load_checkpoint(checkpoint_path, device=device)
        model.eval()
        with torch.inference_mode():
            for length in (1, 48):
                sample = torch.zeros((1, length, 3, 128, 256), device=device)
                output = model(sample)
                if isinstance(output, dict):
                    output = output["emb"]
                if tuple(output.shape) != (1, 256):
                    failures.append(
                        f"Checkpoint T={length} output shape is {tuple(output.shape)}, "
                        "expected (1, 256)."
                    )
                if not torch.isfinite(output).all():
                    failures.append(
                        f"Checkpoint T={length} output contains non-finite values."
                    )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as exc:
        failures.append(f"Checkpoint T=1/T=48 forward validation failed: {exc}")


def _validate_manifest(
    failures: list[str], yaml_values: dict[Path, dict[str, Any]]
) -> None:
    manifest_path = ROOT / "configs" / "experiments" / "manifest.yaml"
    manifest = yaml_values.get(manifest_path, {}).get("experiments", {})
    required_sections = {
        "main_result",
        "history_construction",
        "reliability_diagnostics",
        "cumulative_ablation",
        "leave_one_out",
        "tracker_sensitivity",
        "accuracy_idsw_operating_points",
        "detection_thresholds",
        "detection_dropout",
        "external_validation",
        "internal_scene_analysis",
        "paired_statistics",
        "temporal_length",
        "encoder_comparison",
        "cached_efficiency",
        "cold_efficiency",
        "write_policy",
    }
    missing = sorted(required_sections - set(manifest))
    if missing:
        failures.append(f"Experiment manifest is missing sections: {missing}")

    for name, item in manifest.items():
        for relative in item.get("configs", []):
            if not (ROOT / relative).is_file():
                failures.append(
                    f"Manifest {name} references missing config: {relative}"
                )
        pattern = item.get("config_glob")
        if pattern and not list(ROOT.glob(pattern)):
            failures.append(f"Manifest {name} glob matches no configs: {pattern}")


def _validate_experiment_semantics(
    failures: list[str], yaml_values: dict[Path, dict[str, Any]]
) -> None:
    temporal_path = (
        ROOT
        / "configs"
        / "experiments"
        / "leave_one_out"
        / "w_o_temporal_appearance.yaml"
    )
    temporal = yaml_values.get(temporal_path, {})
    _require_equal(
        failures,
        "temporal-appearance removal representation",
        temporal.get("representation"),
        "appearance_disabled",
    )
    _require_equal(
        failures,
        "temporal-appearance removal association",
        temporal.get("association", {}).get("appearance"),
        False,
    )
    _require_equal(
        failures,
        "temporal-appearance removal geometry",
        temporal.get("association", {}).get("geometry"),
        "iou",
    )
    _require_equal(
        failures,
        "temporal-appearance removal write policy",
        temporal.get("write_policy"),
        "disabled",
    )

    crowd_path = (
        ROOT
        / "configs"
        / "experiments"
        / "leave_one_out"
        / "w_o_crowd_suppression.yaml"
    )
    crowd = yaml_values.get(crowd_path, {})
    _require_equal(
        failures,
        "crowd-damping removal retains reliable writes",
        crowd.get("write_policy"),
        "reliability_first",
    )
    _require_equal(
        failures,
        "crowd-damping removal appearance cap",
        crowd.get("association", {}).get("crowd_appearance_suppression"),
        False,
    )

    for family in ("lengths", "encoders"):
        for path in sorted((ROOT / "configs" / "experiments" / family).glob("*.yaml")):
            config = yaml_values.get(path, {})
            _require_equal(
                failures,
                f"{path.relative_to(ROOT)} independent seeds",
                config.get("seeds"),
                [0, 1, 2],
            )
            if "{seed}" not in str(config.get("checkpoint", "")):
                failures.append(
                    f"{path.relative_to(ROOT)} must select an independently "
                    "trained checkpoint per seed."
                )

    expected_pool_modes = {
        "gru_l48.yaml": "last",
        "lstm_l48.yaml": "last",
        "mean_l48.yaml": "mean",
        "single_frame.yaml": "single_frame",
        "transformer_l48.yaml": "mean",
    }
    for name, expected in expected_pool_modes.items():
        path = ROOT / "configs" / "models" / name
        actual = yaml_values.get(path, {}).get("model", {}).get("pool_mode")
        _require_equal(failures, f"{name} pool mode", actual, expected)

    tracker_source = (
        ROOT / "fishmambatrack" / "tracking" / "amt_tracker.py"
    ).read_text(encoding="utf-8")
    if "self.cfg.crowd_count_th == 1" not in tracker_source:
        failures.append("K_crowd=1 crowd-off sentinel is not implemented.")

    diagnostic_source = (
        ROOT / "scripts" / "diagnose_history_reliability.py"
    ).read_text(encoding="utf-8")
    if "gt_evaluable_write_count" not in diagnostic_source:
        failures.append(
            "History-write error rates do not expose a GT-evaluable denominator."
        )

    cache_source = (ROOT / "scripts" / "build_embedding_cache.py").read_text(
        encoding="utf-8"
    )
    for fragment in (
        '"checkpoint_sha256"',
        '"dataset_config_sha256"',
        '"tracker_config_sha256"',
    ):
        if fragment not in cache_source:
            failures.append(f"Embedding-cache integrity field is missing: {fragment}")


def _validate_trackeval(failures: list[str]) -> None:
    wrapper = (ROOT / "fishmambatrack" / "runtime" / "official_trackeval.py").read_text(
        encoding="utf-8"
    )
    for fragment in (
        '"HOTA"',
        '"CLEAR"',
        '"Identity"',
        '"THRESHOLD": 0.5',
        '"DO_PREPROC": False',
    ):
        if fragment not in wrapper:
            failures.append(f"Official TrackEval policy is missing: {fragment}")


def main() -> None:
    failures: list[str] = []
    yaml_values = _validate_python_and_yaml(failures)
    _validate_main_config(failures, yaml_values)
    _validate_main_semantics(failures)
    _validate_checkpoint(failures)
    _validate_manifest(failures, yaml_values)
    _validate_experiment_semantics(failures, yaml_values)
    _validate_trackeval(failures)
    if failures:
        raise SystemExit("\n".join(failures))
    print(
        "PASS: AMT main method, seed-0 checkpoint, experiment declarations, "
        "and TrackEval policy are consistent."
    )


if __name__ == "__main__":
    main()
