#!/usr/bin/env python3
"""Compute paired Wilcoxon tests and rank-biserial effects by sequence."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import rankdata, wilcoxon


def _rank_biserial(gains: np.ndarray) -> float:
    nonzero = gains[gains != 0]
    if nonzero.size == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-method", default="AMT")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    with args.input.open(newline="", encoding="utf-8-sig") as handle:
        records = list(csv.DictReader(handle))
    indexed = {(row["method"], row["sequence"]): row for row in records}
    rows: list[dict[str, object]] = []
    for baseline in config["comparisons"]:
        for metric in config["metrics"]:
            reference = np.asarray(
                [
                    float(indexed[(args.reference_method, sequence)][metric])
                    for sequence in config["sequences"]
                ],
                dtype=np.float64,
            )
            comparison = np.asarray(
                [
                    float(indexed[(baseline, sequence)][metric])
                    for sequence in config["sequences"]
                ],
                dtype=np.float64,
            )
            gains = (
                comparison - reference if metric == "IDSW" else reference - comparison
            )
            if np.all(gains == 0):
                statistic, p_value = 0.0, 1.0
            else:
                test = wilcoxon(gains, alternative="two-sided", zero_method="wilcox")
                statistic, p_value = float(test.statistic), float(test.pvalue)
            rows.append(
                {
                    "comparison": f"{args.reference_method} vs {baseline}",
                    "metric": metric,
                    "n_sequences": len(gains),
                    "wilcoxon_statistic": statistic,
                    "p_value_two_sided": p_value,
                    "median_paired_improvement": float(np.median(gains)),
                    "mean_paired_improvement": float(np.mean(gains)),
                    "rank_biserial_effect_size": _rank_biserial(gains),
                    "improvement_direction": (
                        f"{baseline} IDSW - {args.reference_method} IDSW"
                        if metric == "IDSW"
                        else f"{args.reference_method} - {baseline}"
                    ),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
