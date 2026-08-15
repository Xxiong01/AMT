#!/usr/bin/env python3
"""Build a persistent, checkpoint-specific current-frame embedding cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fishmambatrack.models.reid.registry import load_checkpoint  # noqa: E402
from fishmambatrack.runtime.online_amt import (  # noqa: E402
    PACKAGE_ROOT,
    atomic_json,
    build_tracker_config,
    encode_current_frame_detections,
    frame_range,
    group_detections,
    infer_detection_coordinates,
    infer_frame_offset,
    read_mot,
    read_yaml,
    set_determinism,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--tracker-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--minimum-score", type=float, default=0.0)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available; pass --device cpu."
        )
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")
    set_determinism(args.seed)
    dataset = read_yaml(args.dataset_config)
    method = read_yaml(args.tracker_config)
    device = torch.device(args.device)
    model, metadata = load_checkpoint(args.checkpoint, device=device)
    length = int(method["temporal_memory"]["length"])
    checkpoint_length = metadata.get("sequence_length")
    if checkpoint_length is not None and int(checkpoint_length) != length:
        raise RuntimeError(
            f"Checkpoint sequence length {checkpoint_length} does not match "
            f"tracker memory length {length}."
        )
    build_tracker_config(method["tracker"], temporal_memory_length=length)
    root_value = Path(dataset["data_root"])
    data_root = (
        root_value.resolve()
        if root_value.is_absolute()
        else (PACKAGE_ROOT / root_value).resolve()
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "cache_manifest.json"
    manifest = {
        "checkpoint": args.checkpoint.name,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_metadata": metadata,
        "dataset_config": args.dataset_config.name,
        "dataset_config_sha256": sha256_file(args.dataset_config),
        "tracker_config": args.tracker_config.name,
        "tracker_config_sha256": sha256_file(args.tracker_config),
        "minimum_score": float(args.minimum_score),
        "sequences": list(dataset["sequences"]),
        "complete_sequences": [],
    }
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in (
            "checkpoint",
            "checkpoint_sha256",
            "checkpoint_metadata",
            "dataset_config",
            "dataset_config_sha256",
            "tracker_config",
            "tracker_config_sha256",
            "minimum_score",
            "sequences",
        ):
            if existing.get(key) != manifest[key]:
                raise RuntimeError(
                    f"Embedding cache metadata mismatch for {key}; use a new output directory."
                )
        manifest["complete_sequences"] = list(existing.get("complete_sequences", []))
    elif list(args.output_dir.glob("*.pt")):
        raise RuntimeError(
            "Cache files exist without a manifest; use a clean output directory."
        )
    atomic_json(manifest_path, manifest)
    for sequence in dataset["sequences"]:
        destination = args.output_dir / f"{sequence}.pt"
        if sequence in manifest["complete_sequences"] and destination.is_file():
            print(f"cached {sequence}: already complete", flush=True)
            continue
        sequence_dir = data_root / sequence
        gt_path = sequence_dir / dataset["gt_file"]
        full_gt = sequence_dir / dataset.get("full_gt_file", dataset["gt_file"])
        bounds = frame_range(gt_path)
        offset = infer_frame_offset(gt_path, full_gt)
        rows = read_mot(sequence_dir / dataset["det_file"])
        global_coordinates = infer_detection_coordinates(
            [row[0] for row in rows], list(range(bounds[0], bounds[1] + 1)), offset
        )
        grouped = group_detections(rows)
        frames = {}
        for frame in range(bounds[0], bounds[1] + 1):
            source_frame = frame + offset if global_coordinates else frame
            detections = encode_current_frame_detections(
                sequence_dir=sequence_dir,
                image_dir=dataset.get("image_dir", "img1"),
                image_ext=dataset.get("image_ext", ".jpg"),
                local_frame=frame,
                frame_offset=offset,
                rows=grouped.get(source_frame, []),
                model=model,
                device=device,
                batch_size=args.batch_size,
                score_threshold=float(args.minimum_score),
                input_size=method.get("model", {}).get("input_size", [128, 256]),
                crop_pad=float(method.get("model", {}).get("crop_pad", 0.10)),
            )
            frames[frame] = [
                {
                    "tlwh": detection.tlwh,
                    "score": detection.score,
                    "frame_feature": detection.frame_feature,
                    "query_embedding": detection.emb,
                }
                for detection in detections
            ]
        temporary = args.output_dir / f".{sequence}.pt.tmp"
        torch.save(frames, temporary)
        temporary.replace(destination)
        manifest["complete_sequences"].append(sequence)
        atomic_json(manifest_path, manifest)
        print(
            f"cached {sequence}: {sum(len(value) for value in frames.values())} detections",
            flush=True,
        )


if __name__ == "__main__":
    main()
