#!/usr/bin/env python3
"""Evaluate AMT trajectories with official TrackEval."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fishmambatrack.runtime.official_trackeval import (  # noqa: E402
    run_official_trackeval,
)
from fishmambatrack.runtime.online_amt import _copy_gt_for_trackeval  # noqa: E402


def _stage_external_results(
    *,
    output_dir: Path,
    dataset_config: Path,
    mot_results_dir: Path,
    tracker_name: str,
) -> None:
    dataset = yaml.safe_load(dataset_config.read_text(encoding="utf-8"))
    if not isinstance(dataset, dict):
        raise ValueError("Dataset configuration must be a YAML mapping.")
    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_gt_for_trackeval(dataset=dataset, output_dir=output_dir)
    benchmark = str(dataset.get("benchmark", "MFT25"))
    split = str(dataset.get("trackeval_split", "val"))
    destination = (
        output_dir
        / "trackeval_data"
        / "trackers"
        / "mot_challenge"
        / f"{benchmark}-{split}"
        / tracker_name
        / "data"
    )
    destination.mkdir(parents=True, exist_ok=True)
    for sequence in dataset["sequences"]:
        source = mot_results_dir / f"{sequence}.txt"
        if not source.is_file():
            raise FileNotFoundError(f"Missing MOT result for {sequence}: {source}")
        shutil.copy2(source, destination / source.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--trackeval-root", type=Path, required=True)
    parser.add_argument("--tracker-name", default="AMT")
    parser.add_argument(
        "--mot-results-dir",
        type=Path,
        help="Optional directory of <sequence>.txt MOT files to stage before evaluation.",
    )
    args = parser.parse_args()
    if args.mot_results_dir is not None:
        _stage_external_results(
            output_dir=args.output_dir.resolve(),
            dataset_config=args.dataset_config.resolve(),
            mot_results_dir=args.mot_results_dir.resolve(),
            tracker_name=args.tracker_name,
        )
    row = run_official_trackeval(
        output_dir=args.output_dir,
        dataset_config=args.dataset_config,
        trackeval_root=args.trackeval_root,
        tracker_name=args.tracker_name,
    )
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
