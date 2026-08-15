#!/usr/bin/env python3
"""Run seed 0/1/2 for one declared paper experiment."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", type=Path, required=True)
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--checkpoint", type=Path)
    checkpoint_group.add_argument("--checkpoint-template")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--trackeval-root", type=Path, required=True)
    parser.add_argument("--embedding-cache-dir", type=Path)
    args = parser.parse_args()
    for seed in args.seeds:
        checkpoint = (
            args.checkpoint
            if args.checkpoint is not None
            else Path(str(args.checkpoint_template).format(seed=int(seed)))
        )
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_experiment.py"),
            "--experiment-config",
            str(args.experiment_config),
            "--checkpoint",
            str(checkpoint),
            "--seed",
            str(seed),
            "--output-dir",
            str(args.output_dir / f"seed_{seed}"),
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
            "--trackeval-root",
            str(args.trackeval_root),
        ]
        if args.embedding_cache_dir is not None:
            command.extend(["--embedding-cache-dir", str(args.embedding_cache_dir)])
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
