#!/usr/bin/env python3
"""Train one declared encoder for seeds 0, 1, and 2."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    for seed in args.seeds:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "train_temporal_reid.py"),
                "--model-config",
                str(args.model_config),
                "--seed",
                str(seed),
                "--output-dir",
                str(args.output_dir / f"seed_{seed}"),
                "--device",
                args.device,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
