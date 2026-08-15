#!/usr/bin/env python3
"""Evaluate AMT trajectories with official TrackEval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fishmambatrack.runtime.official_trackeval import (  # noqa: E402
    run_official_trackeval,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--trackeval-root", type=Path, required=True)
    args = parser.parse_args()
    row = run_official_trackeval(
        output_dir=args.output_dir,
        dataset_config=args.dataset_config,
        trackeval_root=args.trackeval_root,
    )
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
