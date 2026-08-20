"""Online AMT inference with per-track reliability-gated temporal memory.

The current detection is encoded from its current-frame crop only. Temporal
context exists exclusively in a FIFO owned by each active track. A frame
feature enters that FIFO only when the tracker emits an accepted write event.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import os
import random
import shutil
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import transforms

from fishmambatrack.data.mot.mot_utils import infer_frame_offset_from_full_gt
from fishmambatrack.models.reid.registry import load_checkpoint
from fishmambatrack.tracking.amt_tracker import (
    AMTTracker,
    AMTTrackerConfig,
    Detection,
    FishIoUParams,
    l2_normalize,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def read_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_determinism(seed: int) -> None:
    # Required by PyTorch for deterministic CUDA matrix multiplication on
    # CUDA >= 10.2.  Set it before any CUDA kernels are launched.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def read_mot(path: Path) -> List[Tuple[int, int, np.ndarray, float]]:
    rows: List[Tuple[int, int, np.ndarray, float]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = [
                part.strip()
                for part in line.replace(" ", ",").split(",")
                if part.strip()
            ]
            if len(fields) < 6:
                continue
            frame = int(float(fields[0]))
            identity = int(float(fields[1]))
            box = np.asarray([float(value) for value in fields[2:6]], dtype=np.float32)
            score = float(fields[6]) if len(fields) > 6 else 1.0
            if not np.isfinite(box).all() or not np.isfinite(score):
                continue
            if box[2] <= 0.0 or box[3] <= 0.0:
                continue
            rows.append((frame, identity, box, score))
    return rows


def group_detections(
    rows: Iterable[Tuple[int, int, np.ndarray, float]]
) -> Dict[int, list]:
    grouped: Dict[int, list] = defaultdict(list)
    for frame, identity, box, score in rows:
        grouped[int(frame)].append((int(identity), box, float(score)))
    return dict(grouped)


def frame_range(gt_path: Path) -> Tuple[int, int]:
    frames = [row[0] for row in read_mot(gt_path) if row[3] > 0]
    if not frames:
        raise RuntimeError(f"No valid ground-truth rows: {gt_path}")
    return min(frames), max(frames)


def infer_frame_offset(split_gt: Path, full_gt: Path) -> int:
    if split_gt.resolve() == full_gt.resolve():
        return 0
    return int(infer_frame_offset_from_full_gt(split_gt, full_gt))


def infer_detection_coordinates(
    det_frames: Sequence[int], gt_frames: Sequence[int], offset: int
) -> bool:
    """Return True when detections use full-sequence rather than split indices.

    A full-sequence detection file can cover both train and validation halves.
    In that case the local and offset frame sets overlap by the same amount, so
    overlap alone is ambiguous.  The larger detection-frame maximum resolves
    that case without changing the behavior for validation-only local files.
    """
    det_set, gt_set = set(det_frames), set(gt_frames)
    if not det_set or not gt_set or offset == 0:
        return False
    local_overlap = len(det_set & gt_set)
    global_overlap = len(det_set & {frame + offset for frame in gt_set})
    if global_overlap != local_overlap:
        return global_overlap > local_overlap
    return max(det_set) > max(gt_set)


def crop_tlwh(image: Image.Image, box: np.ndarray, pad: float) -> Image.Image:
    x, y, width, height = [float(value) for value in box]
    x -= pad * width
    y -= pad * height
    width *= 1.0 + 2.0 * pad
    height *= 1.0 + 2.0 * pad
    left = max(0, int(np.floor(x)))
    top = max(0, int(np.floor(y)))
    right = min(image.width, int(np.ceil(x + width)))
    bottom = min(image.height, int(np.ceil(y + height)))
    if right <= left or bottom <= top:
        return Image.new("RGB", (2, 2))
    return image.crop((left, top, right, bottom))


def resize_with_pad(image: Image.Image, size: Sequence[int]) -> Image.Image:
    """Resize a ReID crop to the training resolution.

    The temporal ReID checkpoints were trained with torchvision's exact
    ``Resize((height, width))`` transform.  Preserving the aspect ratio here
    would create a train/evaluation mismatch, so evaluation intentionally uses
    the same direct resize.
    """
    output_height, output_width = int(size[0]), int(size[1])
    return image.resize((output_width, output_height), resample=Image.BILINEAR)


def transform_for() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def _backbone_features(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    features = model.backbone(images)
    if features.ndim == 4:
        features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
    return features


def encode_feature_sequences(
    model: torch.nn.Module, features: torch.Tensor
) -> torch.Tensor:
    """Encode precomputed frame features without invoking the image backbone again."""
    if not hasattr(model, "encode_frame_features"):
        raise TypeError("The ReID model does not expose encode_frame_features().")
    return model.encode_frame_features(features).float()


def build_tracker_config(
    values: Mapping[str, Any], *, temporal_memory_length: int
) -> AMTTrackerConfig:
    values = dict(values)
    fish_values = values.pop("fishiou_params", {}) or {}
    valid_fish = {field.name for field in dataclasses.fields(FishIoUParams)}
    unknown_fish = sorted(set(fish_values) - valid_fish)
    if unknown_fish:
        raise ValueError(f"Unknown FishIoU+ configuration keys: {unknown_fish}")
    fish = FishIoUParams(**fish_values)
    valid = {field.name for field in dataclasses.fields(AMTTrackerConfig)}
    unknown = sorted(set(values) - valid)
    if unknown:
        raise ValueError(f"Unknown tracker configuration keys: {unknown}")
    return AMTTrackerConfig(
        **values,
        temporal_memory_length=int(temporal_memory_length),
        fishiou_params=fish,
    )


def encode_current_frame_detections(
    *,
    sequence_dir: Path,
    image_dir: str,
    image_ext: str,
    local_frame: int,
    frame_offset: int,
    rows: Sequence[Tuple[int, np.ndarray, float]],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    score_threshold: float,
    input_size: Sequence[int],
    crop_pad: float,
) -> List[Detection]:
    """Encode only the detections observed in the current video frame."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if len(input_size) != 2 or int(input_size[0]) <= 0 or int(input_size[1]) <= 0:
        raise ValueError(
            f"input_size must contain two positive integers, got {input_size!r}."
        )
    raw = [row for row in rows if row[2] >= score_threshold]
    if not raw:
        return []
    transform = transform_for()
    image_frame = local_frame + frame_offset
    image_path = sequence_dir / image_dir / f"{image_frame:06d}{image_ext}"
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        tensors = [
            transform(resize_with_pad(crop_tlwh(image, box, crop_pad), input_size))
            for _, box, _ in raw
        ]
    detections: List[Detection] = []
    for start in range(0, len(tensors), batch_size):
        stop = min(len(tensors), start + batch_size)
        images = torch.stack(tensors[start:stop]).to(device, non_blocking=True)
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            enabled=device.type == "cuda",
        ):
            features = _backbone_features(model, images)
            queries = encode_feature_sequences(model, features.unsqueeze(1))
        features_np = features.float().cpu().numpy().astype(np.float32, copy=False)
        queries_np = queries.float().cpu().numpy().astype(np.float32, copy=False)
        for index, (_, box, score) in enumerate(raw[start:stop]):
            query = queries_np[index].copy()
            detections.append(
                Detection(
                    tlwh=box.astype(np.float32),
                    score=float(score),
                    emb=query,
                    temporal_query_emb=query,
                    frame_feature=features_np[index].copy(),
                )
            )
    return detections


def _pad_earliest(history: Sequence[np.ndarray], length: int) -> np.ndarray:
    if not history:
        raise ValueError("Cannot encode an empty track history.")
    values = [np.asarray(value, dtype=np.float32) for value in history[-length:]]
    return np.stack([values[0]] * (length - len(values)) + values, axis=0)


def _set_track_embedding(
    track: object, embedding: np.ndarray, depth: int, *, bank_size: int
) -> None:
    value = l2_normalize(np.asarray(embedding, dtype=np.float32).reshape(1, -1))[0]
    track.emb = value
    if int(bank_size) > 0:
        track.emb_bank = deque([value], maxlen=int(bank_size))
    else:
        track.emb_bank = None
    track.temporal_history_depth = int(depth)


def _box_key(box: np.ndarray, score: float) -> Tuple[float, float, float, float, float]:
    values = [round(float(value), 3) for value in np.asarray(box).reshape(-1)[:4]]
    return (*values, round(float(score), 6))


def write_predictions(
    path: Path, predictions: Mapping[int, Sequence[Tuple[int, np.ndarray]]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for frame in sorted(predictions):
            for track_id, box in predictions[frame]:
                x, y, width, height = [float(value) for value in box]
                handle.write(
                    f"{frame},{track_id},{x:.3f},{y:.3f},{width:.3f},{height:.3f},-1,-1,-1,-1\n"
                )


def _copy_gt_for_trackeval(
    *,
    dataset: Mapping[str, Any],
    output_dir: Path,
) -> None:
    root_value = Path(dataset["data_root"])
    root = (
        root_value.resolve()
        if root_value.is_absolute()
        else (PACKAGE_ROOT / root_value).resolve()
    )
    benchmark = str(dataset.get("benchmark", "MFT25"))
    split = str(dataset.get("trackeval_split", "val"))
    gt_root = (
        output_dir / "trackeval_data" / "gt" / "mot_challenge" / f"{benchmark}-{split}"
    )
    seqmap = (
        output_dir
        / "trackeval_data"
        / "gt"
        / "mot_challenge"
        / "seqmaps"
        / f"{benchmark}-{split}.txt"
    )
    seqmap.parent.mkdir(parents=True, exist_ok=True)
    seqmap.write_text(
        "name\n" + "\n".join(dataset["sequences"]) + "\n", encoding="utf-8"
    )
    for sequence in dataset["sequences"]:
        sequence_dir = root / sequence
        source_gt = sequence_dir / dataset["gt_file"]
        lo, hi = frame_range(source_gt)
        destination = gt_root / sequence
        (destination / "gt").mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_gt, destination / "gt" / "gt.txt")
        values = {
            "name": sequence,
            "imDir": dataset.get("image_dir", "img1"),
            "frameRate": int(dataset.get("frame_rate", 25)),
            "seqLength": hi,
            "imWidth": int(dataset.get("image_width", 1920)),
            "imHeight": int(dataset.get("image_height", 1080)),
            "imExt": dataset.get("image_ext", ".jpg"),
        }
        lines = ["[Sequence]"] + [f"{key}={value}" for key, value in values.items()]
        (destination / "seqinfo.ini").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )


def run_tracking(
    *,
    dataset_config: str | Path,
    tracker_config: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    device_name: str = "cuda",
    batch_size: int = 128,
    seed: int = 0,
) -> Dict[str, Any]:
    set_determinism(seed)
    dataset = read_yaml(dataset_config)
    method = read_yaml(tracker_config)
    output_dir = Path(output_dir).resolve()
    checkpoint = Path(checkpoint).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available; pass --device cpu."
        )
    device = torch.device(device_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = method.get("model", {})
    model, checkpoint_meta = load_checkpoint(
        checkpoint,
        device=device,
    )
    sequence_length = int(method["temporal_memory"]["length"])
    checkpoint_length = checkpoint_meta.get("sequence_length")
    if checkpoint_length is not None and int(checkpoint_length) != sequence_length:
        raise RuntimeError(
            f"Checkpoint sequence length {checkpoint_length} does not match "
            f"tracker memory length {sequence_length}."
        )
    expected_model = str(model_cfg.get("name", "mamba"))
    checkpoint_model = str(checkpoint_meta.get("model_name", expected_model))
    if checkpoint_model != expected_model:
        raise RuntimeError(
            f"Checkpoint model {checkpoint_model!r} does not match "
            f"tracker model {expected_model!r}."
        )
    tracker_cfg = build_tracker_config(
        method["tracker"], temporal_memory_length=sequence_length
    )
    input_size = model_cfg.get("input_size", [128, 256])
    crop_pad = float(model_cfg.get("crop_pad", 0.10))
    score_threshold = float(tracker_cfg.det_low_th)
    data_root_value = Path(dataset["data_root"])
    data_root = (
        data_root_value.resolve()
        if data_root_value.is_absolute()
        else (PACKAGE_ROOT / data_root_value).resolve()
    )
    tracker_data = (
        output_dir
        / "trackeval_data"
        / "trackers"
        / "mot_challenge"
        / f"{dataset.get('benchmark', 'MFT25')}-{dataset.get('trackeval_split', 'val')}"
        / "AMT"
        / "data"
    )
    mot_root = output_dir / "mot_results" / "AMT"
    _copy_gt_for_trackeval(dataset=dataset, output_dir=output_dir)

    sequence_rows: List[dict] = []
    diagnostics: List[dict] = []
    start_all = time.perf_counter()
    for sequence in dataset["sequences"]:
        started = time.perf_counter()
        sequence_dir = data_root / sequence
        gt_path = sequence_dir / dataset["gt_file"]
        full_gt = sequence_dir / dataset.get("full_gt_file", dataset["gt_file"])
        det_path = sequence_dir / dataset["det_file"]
        bounds = frame_range(gt_path)
        offset = infer_frame_offset(gt_path, full_gt)
        gt_frames = list(range(bounds[0], bounds[1] + 1))
        det_rows = read_mot(det_path)
        detections_global = infer_detection_coordinates(
            [row[0] for row in det_rows], gt_frames, offset
        )
        detections_by_frame = group_detections(det_rows)
        tracker = AMTTracker(tracker_cfg)
        predictions: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        event_cursor = 0
        write_depths: List[int] = []
        histories: Dict[int, deque] = {}
        for frame in range(bounds[0], bounds[1] + 1):
            source_frame = frame + offset if detections_global else frame
            detections = encode_current_frame_detections(
                sequence_dir=sequence_dir,
                image_dir=dataset.get("image_dir", "img1"),
                image_ext=dataset.get("image_ext", ".jpg"),
                local_frame=frame,
                frame_offset=offset,
                rows=detections_by_frame.get(source_frame, []),
                model=model,
                device=device,
                batch_size=batch_size,
                score_threshold=score_threshold,
                input_size=input_size,
                crop_pad=crop_pad,
            )
            active = tracker.update(detections, frame_id=frame)
            predictions[frame] = [
                (int(track.track_id), track.tlwh.copy()) for track in active
            ]
            events = tracker.diagnostic_events[event_cursor:]
            event_cursor = len(tracker.diagnostic_events)
            changed: List[int] = []
            tracks_by_id = {int(track.track_id): track for track in tracker.tracks}
            detections_by_key = {
                _box_key(detection.tlwh, detection.score): detection
                for detection in detections
            }
            for event in events:
                if not bool(event.get("update_emb", False)):
                    continue
                track_id = int(event["track_id"])
                detection = detections_by_key.get(
                    _box_key(np.asarray(event["tlwh"], dtype=np.float32), event["score"])
                )
                if detection is None or detection.frame_feature is None:
                    raise KeyError(
                        f"Missing current-frame feature for {sequence} frame {frame}: {event}"
                    )
                history = histories.setdefault(track_id, deque(maxlen=sequence_length))
                history.append(np.asarray(detection.frame_feature, dtype=np.float32))
                event["history_depth_after"] = len(history)
                if track_id not in changed:
                    changed.append(track_id)

            active_ids = {int(track.track_id) for track in tracker.tracks}
            for track_id in list(histories):
                if track_id not in active_ids:
                    histories.pop(track_id, None)

            if changed:
                tracks = {int(track.track_id): track for track in tracker.tracks}
                valid = [
                    track_id
                    for track_id in changed
                    if track_id in tracks and track_id in histories and histories[track_id]
                ]
                if valid:
                    values = np.stack(
                        [
                            _pad_earliest(
                                list(histories[track_id]), sequence_length
                            )
                            for track_id in valid
                        ]
                    )
                    tensor = torch.from_numpy(values).to(device)
                    with torch.no_grad(), torch.autocast(
                        device_type=device.type,
                        enabled=device.type == "cuda",
                    ):
                        embeddings = encode_feature_sequences(model, tensor)
                    for track_id, embedding in zip(
                        valid, embeddings.float().cpu().numpy()
                    ):
                        depth = len(histories[track_id])
                        write_depths.append(depth)
                        _set_track_embedding(
                            tracks[track_id],
                            embedding,
                            depth,
                            bank_size=int(tracker_cfg.emb_bank_size),
                        )

        mot_path = mot_root / f"{sequence}.txt"
        tracker_path = tracker_data / f"{sequence}.txt"
        write_predictions(mot_path, predictions)
        tracker_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mot_path, tracker_path)
        event_path = output_dir / "diagnostic_events" / f"{sequence}.jsonl"
        event_path.parent.mkdir(parents=True, exist_ok=True)
        with event_path.open("w", encoding="utf-8") as handle:
            for event in tracker.diagnostic_events:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
        sequence_rows.append(
            {
                "sequence": sequence,
                "mot_path": str(mot_path),
                "frames": bounds[1] - bounds[0] + 1,
                "seconds": time.perf_counter() - started,
            }
        )
        row = dict(tracker.diagnostics)
        row.update(
            sequence=sequence,
            mean_history_depth=float(np.mean(write_depths)) if write_depths else 0.0,
            full_history_fraction=(
                float(np.mean(np.asarray(write_depths) == sequence_length))
                if write_depths
                else 0.0
            ),
        )
        diagnostics.append(row)

    run_summary = {
        "seed": seed,
        "dataset_config": str(Path(dataset_config).resolve()),
        "tracker_config": str(Path(tracker_config).resolve()),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_meta": checkpoint_meta,
        "method_semantics": {
            "current_detection": "current-frame crop only",
            "track_memory": f"independent per-track FIFO, maximum length {sequence_length}",
            "memory_write": "tracker-accepted events only",
            "reactivation_write": "subject to appearance, FishIoU, and crowd checks",
        },
        "sequences": sequence_rows,
        "total_seconds": time.perf_counter() - start_all,
    }
    atomic_json(output_dir / "run_summary.json", run_summary)
    _write_rows(output_dir / "sequence_runtime.csv", sequence_rows)
    _write_rows(output_dir / "reliability_diagnostics.csv", diagnostics)
    return run_summary


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
