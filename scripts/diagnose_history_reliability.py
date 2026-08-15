#!/usr/bin/env python3
"""Offline GT-only diagnosis of FIFO writes and effective history depth."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fishmambatrack.runtime.online_amt import read_mot  # noqa: E402
from fishmambatrack.tracking.amt_tracker import iou_matrix  # noqa: E402


def _resolve_data_root(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _gt_identity(event: dict, frame_gt: list[tuple[int, np.ndarray]]) -> int | None:
    if not frame_gt:
        return None
    box = np.asarray(event["tlwh"], dtype=np.float32).reshape(1, 4)
    boxes = np.stack([item[1] for item in frame_gt])
    overlap = iou_matrix(box, boxes)[0]
    index = int(overlap.argmax())
    return int(frame_gt[index][0]) if float(overlap[index]) >= 0.5 else None


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    dataset = yaml.safe_load(args.dataset_config.read_text(encoding="utf-8"))
    data_root = _resolve_data_root(dataset["data_root"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    depth_rows = []
    for sequence in dataset["sequences"]:
        gt_by_frame: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
        for frame, identity, box, confidence in read_mot(
            data_root / sequence / dataset["gt_file"]
        ):
            if confidence > 0:
                gt_by_frame[frame].append((identity, box))
        events = [
            json.loads(line)
            for line in (args.run_dir / "diagnostic_events" / f"{sequence}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        track_votes: dict[int, Counter[int]] = defaultdict(Counter)
        labelled = []
        for event in events:
            identity = _gt_identity(event, gt_by_frame.get(int(event["frame"]), []))
            track_id = int(event["track_id"])
            reference = (
                track_votes[track_id].most_common(1)[0][0]
                if track_votes[track_id]
                else identity
            )
            wrong = (
                None
                if identity is None
                else bool(reference is not None and identity != reference)
            )
            event["offline_gt_identity"] = identity
            event["offline_gt_evaluable"] = identity is not None
            event["offline_wrong_observation"] = wrong
            labelled.append(event)
            if (
                bool(event.get("update_emb", False))
                and identity is not None
                and wrong is False
            ):
                track_votes[track_id][identity] += 1

        opportunities = [
            event for event in labelled if event.get("stage") != "new_track"
        ]
        writes = [event for event in labelled if bool(event.get("update_emb", False))]
        matched_writes = [
            event for event in writes if event.get("stage") != "new_track"
        ]
        evaluable_opportunities = [
            event for event in opportunities if event["offline_gt_evaluable"]
        ]
        evaluable_writes = [event for event in writes if event["offline_gt_evaluable"]]
        evaluable_matched_writes = [
            event for event in matched_writes if event["offline_gt_evaluable"]
        ]
        rejected = [
            event for event in opportunities if not bool(event.get("update_emb", False))
        ]
        evaluable_rejected = [
            event for event in rejected if event["offline_gt_evaluable"]
        ]
        wrong_writes = sum(
            event["offline_wrong_observation"] is True for event in evaluable_writes
        )
        wrong_matched_writes = sum(
            event["offline_wrong_observation"] is True
            for event in evaluable_matched_writes
        )
        wrong_rejected = sum(
            event["offline_wrong_observation"] is True for event in evaluable_rejected
        )
        reactivation = [
            event for event in opportunities if event.get("stage") == "reactivation"
        ]
        evaluable_reactivation = [
            event for event in reactivation if event["offline_gt_evaluable"]
        ]
        wrong_reactivation = sum(
            event["offline_wrong_observation"] is True
            for event in evaluable_reactivation
        )
        row = {
            "sequence": sequence,
            "successful_match_opportunities": len(opportunities),
            "actual_writes": len(writes),
            "matched_writes": len(matched_writes),
            "matched_write_rate": _ratio(len(matched_writes), len(opportunities)),
            "write_rate": _ratio(len(matched_writes), len(opportunities)),
            "gt_evaluable_write_count": len(evaluable_writes),
            "stage_rejects": sum(event.get("reason") == "stage" for event in rejected),
            "geometry_rejects": sum(
                event.get("reason") == "geometry" for event in rejected
            ),
            "appearance_rejects": sum(
                event.get("reason") == "appearance" for event in rejected
            ),
            "crowd_rejects": sum(event.get("reason") == "crowd" for event in rejected),
            "wrong_writes": wrong_writes,
            "accepted_write_error_rate": _ratio(wrong_writes, len(evaluable_writes)),
            "eligible_event_error_rate": _ratio(
                wrong_matched_writes, len(evaluable_opportunities)
            ),
            "gt_evaluable_rejected_count": len(evaluable_rejected),
            "rejected_observation_error_rate": _ratio(
                wrong_rejected, len(evaluable_rejected)
            ),
            "reactivation_matches": len(reactivation),
            "gt_evaluable_reactivation_matches": len(evaluable_reactivation),
            "wrong_reactivations": wrong_reactivation,
            "wrong_reactivation_rate": _ratio(
                wrong_reactivation, len(evaluable_reactivation)
            ),
        }
        rows.append(row)
        depths = [
            int(event.get("history_depth_after", 0))
            for event in writes
            if event.get("history_depth_after")
        ]
        bins = {
            "1": lambda value: value == 1,
            "2-7": lambda value: 2 <= value <= 7,
            "8-15": lambda value: 8 <= value <= 15,
            "16-31": lambda value: 16 <= value <= 31,
            "32-47": lambda value: 32 <= value <= 47,
            "48": lambda value: value == 48,
        }
        depth_rows.append(
            {
                "sequence": sequence,
                "queries": len(depths),
                "mean": float(np.mean(depths)) if depths else 0.0,
                "median": float(np.median(depths)) if depths else 0.0,
                "q1": float(np.quantile(depths, 0.25)) if depths else 0.0,
                "q3": float(np.quantile(depths, 0.75)) if depths else 0.0,
                "full_48_fraction": _ratio(
                    sum(value == 48 for value in depths), len(depths)
                ),
                "fixed_earliest_padding_fraction": _ratio(
                    sum(value < 48 for value in depths), len(depths)
                ),
                **{
                    f"depth_{label}": sum(predicate(value) for value in depths)
                    for label, predicate in bins.items()
                },
            }
        )
        with (args.output_dir / f"{sequence}_labelled_events.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for event in labelled:
                handle.write(json.dumps(event, sort_keys=True) + "\n")

    for name, values in (
        ("write_reliability.csv", rows),
        ("history_depth.csv", depth_rows),
    ):
        with (args.output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    (args.output_dir / "DIAGNOSTIC_PROTOCOL.txt").write_text(
        "Ground truth is used only after tracking to label accepted and rejected observations; "
        "it is never read by tracker inference. An event is matched to GT at IoU >= 0.5, and "
        "track identity is defined by the running majority of prior accepted correct writes. "
        "Events without an IoU >= 0.5 GT match are reported as non-evaluable and are excluded "
        "from error-rate denominators.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
