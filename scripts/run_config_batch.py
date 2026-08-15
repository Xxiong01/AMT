#!/usr/bin/env python3
"""Run a declared set of controlled experiment configurations."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-glob", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--trackeval-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()

    configs = sorted(ROOT.glob(args.config_glob))
    if not configs:
        raise FileNotFoundError(f"No experiment configs match {args.config_glob!r}")
    for config_path in configs:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        seeds = (
            args.seeds
            if args.seeds is not None
            else config.get("perturbation_seeds", config.get("seeds", [0]))
        )
        for seed in seeds:
            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_experiment.py"),
                "--experiment-config",
                str(config_path),
                "--seed",
                str(seed),
                "--output-dir",
                str(args.output_dir / config_path.stem / f"seed_{seed}"),
                "--device",
                args.device,
                "--batch-size",
                str(args.batch_size),
                "--trackeval-root",
                str(args.trackeval_root),
            ]
            if args.checkpoint is not None:
                command.extend(["--checkpoint", str(args.checkpoint)])
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
