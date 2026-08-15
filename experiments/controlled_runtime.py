"""Controlled representation experiments; never used by the AMT main entry."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from experiments.controlled_tracker import ControlledAMTTracker
from fishmambatrack.runtime.online_amt import (
    PACKAGE_ROOT,
    _copy_gt_for_trackeval,
    _pad_earliest,
    _set_track_embedding,
    _write_rows,
    atomic_json,
    build_tracker_config,
    encode_current_frame_detections,
    encode_feature_sequences,
    frame_range,
    group_detections,
    infer_detection_coordinates,
    infer_frame_offset,
    read_mot,
    read_yaml,
    set_determinism,
    sha256_file,
    write_predictions,
)
from fishmambatrack.models.reid.registry import load_checkpoint
from fishmambatrack.tracking.amt_tracker import Detection, iou_matrix


def _detection_key(box: np.ndarray, score: float) -> tuple[float, ...]:
    values = [float(value) for value in np.asarray(box, dtype=np.float32)]
    return tuple(round(value, 5) for value in values + [float(score)])


def _detections_from_cache(
    records: Sequence[Mapping[str, Any]],
    rows: Sequence[Tuple[int, np.ndarray, float]],
    score_threshold: float,
) -> list[Detection]:
    """Select cached features using the current controlled detection file."""
    available = Counter(
        _detection_key(box, score)
        for _, box, score in rows
        if float(score) >= score_threshold
    )
    detections: list[Detection] = []
    for record in records:
        key = _detection_key(record["tlwh"], float(record["score"]))
        if available[key] <= 0:
            continue
        available[key] -= 1
        query = np.asarray(record["query_embedding"], dtype=np.float32)
        detections.append(
            Detection(
                tlwh=np.asarray(record["tlwh"], dtype=np.float32),
                score=float(record["score"]),
                emb=query,
                temporal_query_emb=query,
                frame_feature=np.asarray(record["frame_feature"], dtype=np.float32),
            )
        )
    if sum(available.values()) != 0:
        raise RuntimeError(
            "Embedding cache does not cover the active detection file; rebuild it "
            "with --minimum-score 0.0 for this checkpoint."
        )
    return detections


def _detections_without_appearance(
    rows: Sequence[Tuple[int, np.ndarray, float]], score_threshold: float
) -> list[Detection]:
    return [
        Detection(tlwh=np.asarray(box, dtype=np.float32), score=float(score))
        for _, box, score in rows
        if float(score) >= score_threshold
    ]


def _encode_alternative_queries(
    detections: list[Detection],
    *,
    mode: str,
    length: int,
    model: torch.nn.Module,
    device: torch.device,
    previous: Sequence[Tuple[np.ndarray, list[np.ndarray]]],
    history_iou_threshold: float,
    fallback: bool,
    fallback_threshold: float,
) -> tuple[list[Tuple[np.ndarray, list[np.ndarray]]], int]:
    if not detections or mode in {
        "appearance_disabled",
        "single_frame",
        "online_per_track_fifo",
    }:
        return [], 0
    current_features = [
        np.asarray(det.frame_feature, dtype=np.float32) for det in detections
    ]
    histories: list[list[np.ndarray]] = []
    if mode == "repeat_current_frame":
        histories = [[feature] * length for feature in current_features]
    elif mode == "detection_linked_history":
        current_boxes = np.stack([det.tlwh for det in detections])
        previous_boxes = (
            np.stack([box for box, _ in previous])
            if previous
            else np.zeros((0, 4), dtype=np.float32)
        )
        overlap = iou_matrix(current_boxes, previous_boxes)
        for index, feature in enumerate(current_features):
            if previous and float(overlap[index].max()) >= history_iou_threshold:
                parent = int(overlap[index].argmax())
                histories.append((previous[parent][1] + [feature])[-length:])
            else:
                histories.append([feature])
    else:
        raise ValueError(f"Unsupported controlled representation: {mode}")

    values = np.stack([_pad_earliest(history, length) for history in histories])
    with torch.no_grad(), torch.autocast(
        device_type=device.type, enabled=device.type == "cuda"
    ):
        embeddings = (
            encode_feature_sequences(model, torch.from_numpy(values).to(device))
            .cpu()
            .numpy()
        )
    fallback_count = 0
    for detection, embedding in zip(detections, embeddings):
        current = np.asarray(detection.emb, dtype=np.float32)
        selected = np.asarray(embedding, dtype=np.float32)
        if fallback:
            cosine = float(
                current
                @ selected
                / ((np.linalg.norm(current) * np.linalg.norm(selected)) + 1e-12)
            )
            if cosine < fallback_threshold:
                selected = current
                fallback_count += 1
        detection.emb = selected
        detection.temporal_query_emb = selected
    state = [
        (detection.tlwh.copy(), list(history))
        for detection, history in zip(detections, histories)
    ]
    return state, fallback_count


def run_controlled_tracking(
    *,
    experiment: Mapping[str, Any],
    dataset_config: str | Path,
    tracker_config: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    device_name: str = "cuda",
    batch_size: int = 128,
    seed: int = 0,
    embedding_cache_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """Run one explicitly labelled comparison outside the main AMT path."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available; pass --device cpu."
        )
    set_determinism(seed)
    dataset = read_yaml(dataset_config)
    method = read_yaml(tracker_config)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    model, checkpoint_meta = load_checkpoint(checkpoint, device=device)
    length = int(experiment.get("sequence_length", method["temporal_memory"]["length"]))
    checkpoint_length = checkpoint_meta.get("sequence_length")
    if checkpoint_length is not None and int(checkpoint_length) != length:
        raise RuntimeError(
            f"Checkpoint sequence length {checkpoint_length} does not match "
            f"experiment sequence length {length}."
        )
    tracker_values = dict(method["tracker"])
    tracker_values.update(dict(experiment.get("tracker_overrides", {})))
    tracker_cfg = build_tracker_config(tracker_values, temporal_memory_length=length)
    mode = str(experiment.get("representation", "online_per_track_fifo"))
    valid_modes = {
        "appearance_disabled",
        "single_frame",
        "online_per_track_fifo",
        "repeat_current_frame",
        "detection_linked_history",
    }
    if mode not in valid_modes:
        raise ValueError(f"Unsupported controlled representation: {mode}")
    data_root_value = Path(dataset["data_root"])
    data_root = (
        data_root_value.resolve()
        if data_root_value.is_absolute()
        else (PACKAGE_ROOT / data_root_value).resolve()
    )
    benchmark = str(dataset.get("benchmark", "MFT25"))
    split = str(dataset.get("trackeval_split", "val"))
    tracker_data = (
        output_dir
        / "trackeval_data"
        / "trackers"
        / "mot_challenge"
        / f"{benchmark}-{split}"
        / "AMT"
        / "data"
    )
    mot_root = output_dir / "mot_results" / "AMT"
    _copy_gt_for_trackeval(dataset=dataset, output_dir=output_dir)
    sequence_rows: List[dict] = []
    diagnostics: List[dict] = []
    timing_rows: List[dict] = []
    started_all = time.perf_counter()
    cache_root = Path(embedding_cache_dir).resolve() if embedding_cache_dir else None
    if cache_root is not None:
        cache_manifest_path = cache_root / "cache_manifest.json"
        if not cache_manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing embedding cache manifest: {cache_manifest_path}"
            )
        cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        if cache_manifest.get("checkpoint_metadata") != checkpoint_meta:
            raise RuntimeError(
                "Embedding cache checkpoint metadata does not match the loaded checkpoint."
            )
        expected_hashes = {
            "checkpoint_sha256": sha256_file(checkpoint),
            "tracker_config_sha256": sha256_file(tracker_config),
        }
        if float(experiment.get("detection", {}).get("dropout_ratio", 0.0)) <= 0.0:
            expected_hashes["dataset_config_sha256"] = sha256_file(dataset_config)
        for key, expected in expected_hashes.items():
            if cache_manifest.get(key) != expected:
                raise RuntimeError(
                    f"Embedding cache {key} does not match the active run."
                )
        if float(cache_manifest.get("minimum_score", 1.0)) > float(
            tracker_cfg.det_low_th
        ):
            raise RuntimeError(
                "Embedding cache minimum score is above the experiment detection threshold."
            )
    for sequence in dataset["sequences"]:
        started = time.perf_counter()
        sequence_dir = data_root / sequence
        gt_path = sequence_dir / dataset["gt_file"]
        full_gt = sequence_dir / dataset.get("full_gt_file", dataset["gt_file"])
        bounds = frame_range(gt_path)
        offset = infer_frame_offset(gt_path, full_gt)
        det_rows = read_mot(sequence_dir / dataset["det_file"])
        detections_global = infer_detection_coordinates(
            [row[0] for row in det_rows],
            list(range(bounds[0], bounds[1] + 1)),
            offset,
        )
        detections_by_frame = group_detections(det_rows)
        cached_frames = None
        if cache_root is not None:
            cache_path = cache_root / f"{sequence}.pt"
            if not cache_path.is_file():
                raise FileNotFoundError(
                    f"Missing checkpoint-specific embedding cache: {cache_path}"
                )
            cached_frames = torch.load(
                cache_path, map_location="cpu", weights_only=False
            )
        tracker = ControlledAMTTracker(tracker_cfg, experiment)
        predictions: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        event_cursor = 0
        previous_detection_history: list[Tuple[np.ndarray, list[np.ndarray]]] = []
        fallback_count = 0
        query_count = 0
        crop_backbone_seconds = 0.0
        history_query_seconds = 0.0
        association_seconds = 0.0
        fifo_mamba_seconds = 0.0
        crop_count = 0
        mamba_token_count = 0
        for frame in range(bounds[0], bounds[1] + 1):
            source_frame = frame + offset if detections_global else frame
            phase_started = time.perf_counter()
            active_rows = detections_by_frame.get(source_frame, [])
            if mode == "appearance_disabled":
                detections = _detections_without_appearance(
                    active_rows, float(tracker_cfg.det_low_th)
                )
            elif cached_frames is None:
                detections = encode_current_frame_detections(
                    sequence_dir=sequence_dir,
                    image_dir=dataset.get("image_dir", "img1"),
                    image_ext=dataset.get("image_ext", ".jpg"),
                    local_frame=frame,
                    frame_offset=offset,
                    rows=active_rows,
                    model=model,
                    device=device,
                    batch_size=batch_size,
                    score_threshold=float(tracker_cfg.det_low_th),
                    input_size=method.get("model", {}).get("input_size", [128, 256]),
                    crop_pad=float(method.get("model", {}).get("crop_pad", 0.10)),
                )
            else:
                detections = _detections_from_cache(
                    cached_frames.get(frame, []),
                    active_rows,
                    float(tracker_cfg.det_low_th),
                )
            crop_backbone_seconds += time.perf_counter() - phase_started
            crop_count += len(detections)
            phase_started = time.perf_counter()
            previous_detection_history, triggered = _encode_alternative_queries(
                detections,
                mode=mode,
                length=length,
                model=model,
                device=device,
                previous=previous_detection_history,
                history_iou_threshold=float(
                    experiment.get("history_iou_threshold", 0.20)
                ),
                fallback=bool(experiment.get("single_frame_fallback", False)),
                fallback_threshold=float(
                    experiment.get("fallback_cosine_threshold", 0.95)
                ),
            )
            fallback_count += triggered
            if bool(experiment.get("single_frame_fallback", False)):
                query_count += len(detections)
            if mode in {"repeat_current_frame", "detection_linked_history"}:
                mamba_token_count += len(detections) * length
            history_query_seconds += time.perf_counter() - phase_started
            phase_started = time.perf_counter()
            active = tracker.update(detections, frame_id=frame)
            association_seconds += time.perf_counter() - phase_started
            predictions[frame] = [
                (int(track.track_id), track.tlwh.copy()) for track in active
            ]
            events = tracker.diagnostic_events[event_cursor:]
            event_cursor = len(tracker.diagnostic_events)
            tracks_by_id = {int(track.track_id): track for track in tracker.tracks}
            for event in events:
                track_id = int(event["track_id"])
                if bool(event.get("update_emb", False)) and track_id in tracks_by_id:
                    event["history_depth_after"] = len(
                        tracks_by_id[track_id].temporal_history
                    )
            changed = {
                int(event["track_id"])
                for event in events
                if bool(event.get("update_emb", False))
            }
            if mode == "online_per_track_fifo" and changed:
                phase_started = time.perf_counter()
                tracks = {int(track.track_id): track for track in tracker.tracks}
                valid = [
                    track_id
                    for track_id in changed
                    if track_id in tracks and tracks[track_id].temporal_history
                ]
                if valid:
                    mamba_token_count += len(valid) * length
                    values = np.stack(
                        [
                            _pad_earliest(
                                list(tracks[track_id].temporal_history), length
                            )
                            for track_id in valid
                        ]
                    )
                    with torch.no_grad(), torch.autocast(
                        device_type=device.type, enabled=device.type == "cuda"
                    ):
                        embeddings = (
                            encode_feature_sequences(
                                model, torch.from_numpy(values).to(device)
                            )
                            .cpu()
                            .numpy()
                        )
                    for track_id, embedding in zip(valid, embeddings):
                        _set_track_embedding(
                            tracks[track_id],
                            embedding,
                            len(tracks[track_id].temporal_history),
                        )
                fifo_mamba_seconds += time.perf_counter() - phase_started
        mot_path = mot_root / f"{sequence}.txt"
        tracker_path = tracker_data / f"{sequence}.txt"
        write_predictions(mot_path, predictions)
        tracker_path.parent.mkdir(parents=True, exist_ok=True)
        tracker_path.write_bytes(mot_path.read_bytes())
        event_path = output_dir / "diagnostic_events" / f"{sequence}.jsonl"
        event_path.parent.mkdir(parents=True, exist_ok=True)
        with event_path.open("w", encoding="utf-8") as handle:
            for event in tracker.diagnostic_events:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
        sequence_rows.append(
            {
                "sequence": sequence,
                "frames": bounds[1] - bounds[0] + 1,
                "seconds": time.perf_counter() - started,
            }
        )
        row = dict(tracker.diagnostics)
        row.update(
            {
                "sequence": sequence,
                "fallback_checks": query_count,
                "fallback_triggers": fallback_count,
            }
        )
        diagnostics.append(row)
        timing_rows.append(
            {
                "sequence": sequence,
                "crop_preprocess_resnet_seconds": crop_backbone_seconds,
                "detection_history_seconds": history_query_seconds,
                "fifo_mamba_seconds": fifo_mamba_seconds,
                "association_including_reactivation_seconds": association_seconds,
                "crop_count": crop_count,
                "mamba_token_count": mamba_token_count,
            }
        )
    summary = {
        "experiment_id": experiment.get("experiment_id"),
        "controlled_comparison_only": True,
        "seed": seed,
        "checkpoint_metadata": checkpoint_meta,
        "representation": mode,
        "total_seconds": time.perf_counter() - started_all,
        "sequences": sequence_rows,
    }
    atomic_json(output_dir / "run_summary.json", summary)
    _write_rows(output_dir / "sequence_runtime.csv", sequence_rows)
    _write_rows(output_dir / "diagnostics.csv", diagnostics)
    _write_rows(output_dir / "timing_breakdown.csv", timing_rows)
    return summary
