#!/usr/bin/env python3
"""Evaluate every completed tracking directory below one experiment root."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--trackeval-root", type=Path, required=True)
    args = parser.parse_args()
    runs = sorted(
        path.parent
        for path in args.runs_root.rglob("trackeval_data")
        if (path / "trackers" / "mot_challenge").is_dir()
    )
    for run in runs:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "evaluate_trackeval.py"),
                "--output-dir",
                str(run),
                "--dataset-config",
                str(args.dataset_config),
                "--trackeval-root",
                str(args.trackeval_root),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
