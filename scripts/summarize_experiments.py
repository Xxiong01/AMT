#!/usr/bin/env python3
"""Aggregate official TrackEval CSV files, including sample standard deviation."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

METRICS = ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW", "FP", "FN", "Frag")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    grouped: dict[str, list[dict[str, str]]] = {}
    for path in sorted(args.runs_root.rglob("official_trackeval_metrics.csv")):
        with path.open(newline="", encoding="utf-8-sig") as handle:
            row = next(csv.DictReader(handle))
        experiment = (
            path.parent.parent.name
            if path.parent.name.startswith("seed_")
            else path.parent.name
        )
        grouped.setdefault(experiment, []).append(row)
    output = []
    for experiment, rows in grouped.items():
        result = {"experiment": experiment, "runs": len(rows)}
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            result[f"{metric}_mean"] = statistics.mean(values)
            result[f"{metric}_sample_sd"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        output.append(result)
    if not output:
        raise RuntimeError(
            f"No official_trackeval_metrics.csv files below {args.runs_root}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)


if __name__ == "__main__":
    main()
