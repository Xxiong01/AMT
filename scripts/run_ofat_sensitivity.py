#!/usr/bin/env python3
"""Expand and run the declared one-factor-at-a-time sensitivity protocol."""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _tag(value: object) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trackeval-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    source = yaml.safe_load(args.experiment_config.read_text(encoding="utf-8"))
    factors = source.pop("one_factor_at_a_time")
    seeds = source.get("seeds", [0])
    for parameter, values in factors.items():
        for value in values:
            config = copy.deepcopy(source)
            config["experiment_id"] = f"ofat_{parameter}_{_tag(value)}"
            config["tracker_overrides"] = {parameter: value}
            run_root = args.output_dir / parameter / _tag(value)
            run_root.mkdir(parents=True, exist_ok=True)
            expanded = run_root / "expanded_experiment.yaml"
            expanded.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            for seed in seeds:
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / "run_experiment.py"),
                        "--experiment-config",
                        str(expanded),
                        "--checkpoint",
                        str(args.checkpoint),
                        "--seed",
                        str(seed),
                        "--output-dir",
                        str(run_root / f"seed_{seed}"),
                        "--trackeval-root",
                        str(args.trackeval_root),
                        "--device",
                        args.device,
                        "--batch-size",
                        str(args.batch_size),
                    ],
                    check=True,
                )


if __name__ == "__main__":
    main()
