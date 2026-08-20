"""Official TrackEval invocation and machine-readable metric export."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import yaml

TRACKEVAL_COMMIT = "12c8791b303e0a0b50f753af204249e622d0281a"


def _validate_tracker_name(value: str) -> str:
    name = str(value).strip()
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError(f"Invalid tracker name: {value!r}")
    return name


def _verify_trackeval_revision(root: Path) -> None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root.resolve()), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "TrackEval must be a Git checkout so that the evaluation revision can "
            "be verified. Follow REPRODUCIBILITY.md and check out the pinned commit."
        ) from exc
    revision = result.stdout.strip().lower()
    if revision != TRACKEVAL_COMMIT:
        raise RuntimeError(
            f"TrackEval revision mismatch: expected {TRACKEVAL_COMMIT}, found "
            f"{revision or '<empty>'}."
        )


def _load_trackeval(root: Path):
    _verify_trackeval_revision(root)
    sys.path.insert(0, str(root.resolve()))
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]
    try:
        import trackeval  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Official TrackEval is required. Clone JonathonLuiten/TrackEval "
            "and pass its directory with --trackeval-root."
        ) from exc
    return trackeval


def _summary(output_dir: Path, tracker_name: str) -> Dict[str, str]:
    source = (
        output_dir
        / "trackeval_raw"
        / tracker_name
        / "pedestrian_summary.txt"
    )
    lines = [
        line.split()
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    row = dict(zip(lines[0], lines[1]))
    return {
        "method": tracker_name,
        "HOTA": row.get("HOTA", ""),
        "DetA": row.get("DetA", ""),
        "AssA": row.get("AssA", ""),
        "IDF1": row.get("IDF1", ""),
        "MOTA": row.get("MOTA", ""),
        "IDSW": row.get("IDSW", ""),
        "FP": row.get("CLR_FP", ""),
        "FN": row.get("CLR_FN", ""),
        "Frag": row.get("Frag", ""),
    }


def _write_per_sequence(output_dir: Path, tracker_name: str) -> None:
    source = (
        output_dir
        / "trackeval_raw"
        / tracker_name
        / "pedestrian_detailed.csv"
    )
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    fields = (
        "sequence",
        "HOTA",
        "DetA",
        "AssA",
        "IDF1",
        "MOTA",
        "IDSW",
        "FP",
        "FN",
        "Frag",
    )
    destination = output_dir / "official_trackeval_per_sequence.csv"
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            sequence = row.get("seq", "")
            if not sequence or sequence.upper() == "COMBINED_SEQ":
                continue
            writer.writerow(
                {
                    "sequence": sequence,
                    "HOTA": f"{100.0 * float(row['HOTA___AUC']):.6f}",
                    "DetA": f"{100.0 * float(row['DetA___AUC']):.6f}",
                    "AssA": f"{100.0 * float(row['AssA___AUC']):.6f}",
                    "IDF1": f"{100.0 * float(row['IDF1']):.6f}",
                    "MOTA": f"{100.0 * float(row['MOTA']):.6f}",
                    "IDSW": int(float(row["IDSW"])),
                    "FP": int(float(row["CLR_FP"])),
                    "FN": int(float(row["CLR_FN"])),
                    "Frag": int(float(row["Frag"])),
                }
            )


def run_official_trackeval(
    *,
    output_dir: str | Path,
    dataset_config: str | Path,
    trackeval_root: str | Path,
    tracker_name: str = "AMT",
) -> Dict[str, str]:
    output_dir = Path(output_dir).resolve()
    tracker_name = _validate_tracker_name(tracker_name)
    dataset = yaml.safe_load(Path(dataset_config).read_text(encoding="utf-8"))
    trackeval = _load_trackeval(Path(trackeval_root))
    benchmark = str(dataset["benchmark"])
    split = str(dataset["trackeval_split"])
    data_root = output_dir / "trackeval_data"

    evaluator_config = trackeval.Evaluator.get_default_eval_config()
    evaluator_config.update(
        {
            "USE_PARALLEL": False,
            "BREAK_ON_ERROR": True,
            "PRINT_RESULTS": True,
            "PRINT_ONLY_COMBINED": False,
            "OUTPUT_SUMMARY": True,
            "OUTPUT_DETAILED": True,
            "PLOT_CURVES": False,
        }
    )
    dataset_values = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_values.update(
        {
            "GT_FOLDER": str(data_root / "gt" / "mot_challenge"),
            "TRACKERS_FOLDER": str(data_root / "trackers" / "mot_challenge"),
            "OUTPUT_FOLDER": str(output_dir / "trackeval_raw"),
            "TRACKERS_TO_EVAL": [tracker_name],
            "CLASSES_TO_EVAL": ["pedestrian"],
            "BENCHMARK": benchmark,
            "SPLIT_TO_EVAL": split,
            "INPUT_AS_ZIP": False,
            "DO_PREPROC": False,
            "TRACKER_SUB_FOLDER": "data",
            "OUTPUT_SUB_FOLDER": "",
            "SEQMAP_FILE": str(
                data_root
                / "gt"
                / "mot_challenge"
                / "seqmaps"
                / f"{benchmark}-{split}.txt"
            ),
            "SKIP_SPLIT_FOL": False,
        }
    )
    metric_values = {"METRICS": ["HOTA", "CLEAR", "Identity"], "THRESHOLD": 0.5}
    evaluator = trackeval.Evaluator(evaluator_config)
    evaluator.evaluate(
        [trackeval.datasets.MotChallenge2DBox(dataset_values)],
        [
            trackeval.metrics.HOTA(metric_values),
            trackeval.metrics.CLEAR(metric_values),
            trackeval.metrics.Identity(metric_values),
        ],
    )

    row = _summary(output_dir, tracker_name)
    _write_per_sequence(output_dir, tracker_name)
    with (output_dir / "official_trackeval_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    (output_dir / "evaluation_protocol.txt").write_text(
        "Official TrackEval; HOTA, CLEAR, Identity; similarity threshold 0.5; "
        "DO_PREPROC=False.\n",
        encoding="utf-8",
    )
    return row
