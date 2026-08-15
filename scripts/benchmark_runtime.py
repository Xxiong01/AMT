#!/usr/bin/env python3
"""Three-run cached or cold end-to-end runtime benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_monitored(command: list[str]) -> tuple[float, float, float]:
    import psutil

    process = subprocess.Popen(command)
    peak_ram = 0
    peak_vram = 0.0
    started = time.perf_counter()
    while process.poll() is None:
        try:
            members = [psutil.Process(process.pid)] + psutil.Process(
                process.pid
            ).children(recursive=True)
            pids = {member.pid for member in members}
            peak_ram = max(
                peak_ram,
                sum(
                    member.memory_info().rss
                    for member in members
                    if member.is_running()
                ),
            )
            query = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            used = 0.0
            for line in query.stdout.splitlines():
                fields = [part.strip() for part in line.split(",")]
                if len(fields) == 2 and int(fields[0]) in pids:
                    used += float(fields[1])
            peak_vram = max(peak_vram, used)
        except (psutil.Error, ValueError, FileNotFoundError):
            pass
        time.sleep(0.25)
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)
    return time.perf_counter() - started, peak_ram / (1024.0**2), peak_vram


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("cached", "cold"), required=True)
    parser.add_argument("--embedding-cache-dir", type=Path)
    parser.add_argument("--trackeval-root", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.mode == "cached" and not args.embedding_cache_dir:
        raise ValueError("--embedding-cache-dir is required in cached mode")
    if args.mode == "cold" and args.embedding_cache_dir is not None:
        raise ValueError("--embedding-cache-dir must not be used in cold mode")
    if args.repetitions <= 0:
        raise ValueError("--repetitions must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for repeat in range(args.repetitions):
        run_dir = args.output_dir / f"run_{repeat + 1}"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_experiment.py"),
            "--experiment-config",
            str(args.experiment_config),
            "--checkpoint",
            str(args.checkpoint),
            "--seed",
            "0",
            "--output-dir",
            str(run_dir),
            "--device",
            args.device,
            "--trackeval-root",
            str(args.trackeval_root),
        ]
        if args.embedding_cache_dir:
            command += ["--embedding-cache-dir", str(args.embedding_cache_dir)]
        wall, peak_ram, peak_vram = _run_monitored(command)
        summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
        with (run_dir / "timing_breakdown.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            timing = list(csv.DictReader(handle))
        with (run_dir / "official_trackeval_metrics.csv").open(
            newline="", encoding="utf-8-sig"
        ) as handle:
            metric = next(csv.DictReader(handle))
        frames = sum(int(row["frames"]) for row in summary["sequences"])
        trajectory = float(summary["total_seconds"])
        rows.append(
            {
                "repeat": repeat + 1,
                "mode": args.mode,
                "trajectory_generation_seconds": trajectory,
                "crop_preprocess_resnet_seconds": sum(
                    float(row["crop_preprocess_resnet_seconds"]) for row in timing
                ),
                "detection_history_seconds": sum(
                    float(row["detection_history_seconds"]) for row in timing
                ),
                "fifo_mamba_seconds": sum(
                    float(row["fifo_mamba_seconds"]) for row in timing
                ),
                "association_including_reactivation_seconds": sum(
                    float(row["association_including_reactivation_seconds"])
                    for row in timing
                ),
                "crop_count": sum(int(row["crop_count"]) for row in timing),
                "mamba_token_count": sum(
                    int(row["mamba_token_count"]) for row in timing
                ),
                "peak_ram_mib": peak_ram,
                "peak_vram_mib": peak_vram,
                "fps": frames / trajectory,
                "ms_per_frame": 1000.0 * trajectory / frames,
                "trackeval_and_process_overhead_seconds": wall - trajectory,
                "HOTA": metric["HOTA"],
                "IDF1": metric["IDF1"],
                "IDSW": metric["IDSW"],
            }
        )
    with (args.output_dir / "runtime_repetitions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregate = []
    for field in rows[0]:
        if field in {"repeat", "mode", "HOTA", "IDF1", "IDSW"}:
            continue
        values = [float(row[field]) for row in rows]
        aggregate.append(
            {
                "metric": field,
                "mean": statistics.mean(values),
                "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
        )
    with (args.output_dir / "runtime_mean_sample_sd.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)


if __name__ == "__main__":
    main()
