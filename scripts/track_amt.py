#!/usr/bin/env python3
"""Generate trajectories with online AMT and a declared frozen configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKER_CONFIG = ROOT / "configs" / "tracker" / "amt_l48.yaml"
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "best.pt"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fishmambatrack.runtime.online_amt import run_tracking  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--tracker-config", type=Path, default=DEFAULT_TRACKER_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    result = run_tracking(
        dataset_config=args.dataset_config,
        tracker_config=args.tracker_config,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        device_name=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
