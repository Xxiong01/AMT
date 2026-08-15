#!/usr/bin/env python3
"""Run one controlled paper experiment from a declarative YAML file."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.controlled_runtime import run_controlled_tracking  # noqa: E402
from fishmambatrack.runtime.official_trackeval import (  # noqa: E402
    run_official_trackeval,
)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _resolve_for_seed(path: str | Path, seed: int) -> Path:
    try:
        rendered = str(path).format(seed=int(seed))
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Invalid checkpoint template: {path}") from exc
    return _resolve(rendered)


def _link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return
    try:
        os.symlink(source, destination, target_is_directory=source.is_dir())
    except OSError:
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)


def _perturbed_dataset(
    dataset_path: Path,
    experiment: Dict[str, Any],
    output_dir: Path,
    seed: int,
) -> Path:
    detection = dict(experiment.get("detection", {}))
    dropout = float(detection.get("dropout_ratio", 0.0))
    if dropout <= 0:
        return dataset_path
    dataset = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
    source_root = _resolve(dataset["data_root"])
    target_root = output_dir / "controlled_detection_input"
    rng = random.Random(seed)
    for sequence in dataset["sequences"]:
        source_sequence = source_root / sequence
        target_sequence = target_root / sequence
        _link(
            source_sequence / dataset.get("image_dir", "img1"),
            target_sequence / dataset.get("image_dir", "img1"),
        )
        _link(source_sequence / "gt", target_sequence / "gt")
        source_det = source_sequence / dataset["det_file"]
        target_det = target_sequence / dataset["det_file"]
        target_det.parent.mkdir(parents=True, exist_ok=True)
        with source_det.open(encoding="utf-8") as source, target_det.open(
            "w", encoding="utf-8"
        ) as target:
            for line in source:
                if line.strip() and rng.random() >= dropout:
                    target.write(line)
    dataset["data_root"] = str(target_root)
    adapted = output_dir / "dataset_config.yaml"
    adapted.write_text(yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8")
    return adapted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--trackeval-root", type=Path)
    parser.add_argument("--embedding-cache-dir", type=Path)
    args = parser.parse_args()

    experiment = yaml.safe_load(args.experiment_config.read_text(encoding="utf-8"))
    if not isinstance(experiment, dict):
        raise ValueError("Experiment configuration must be a YAML mapping.")
    if not experiment.get("controlled_comparison_only", False):
        raise ValueError(
            "Experiment configs must declare controlled_comparison_only: true"
        )
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    output = args.output_dir.resolve()
    dataset_config = _resolve(experiment["dataset_config"])
    tracker_config = _resolve(experiment["tracker_config"])
    checkpoint_source = args.checkpoint if args.checkpoint else experiment["checkpoint"]
    checkpoint = _resolve_for_seed(checkpoint_source, args.seed)
    for label, path in (
        ("dataset configuration", dataset_config),
        ("tracker configuration", tracker_config),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found for seed {args.seed}: {checkpoint}"
        )
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.experiment_config, output / "experiment_config.yaml")
    dataset_config = _perturbed_dataset(dataset_config, experiment, output, args.seed)

    detection = dict(experiment.get("detection", {}))
    if "score_threshold" in detection:
        experiment.setdefault("tracker_overrides", {})["det_low_th"] = float(
            detection["score_threshold"]
        )
    summary = run_controlled_tracking(
        experiment=experiment,
        dataset_config=dataset_config,
        tracker_config=tracker_config,
        checkpoint=checkpoint,
        output_dir=output,
        device_name=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
        embedding_cache_dir=args.embedding_cache_dir,
    )
    if args.trackeval_root:
        summary["official_trackeval"] = run_official_trackeval(
            output_dir=output,
            dataset_config=dataset_config,
            trackeval_root=args.trackeval_root,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
