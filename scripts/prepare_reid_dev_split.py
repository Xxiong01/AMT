#!/usr/bin/env python3
"""Create the fixed Train-derived development split used for AMT-L48."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

DEV_FRACTION = 0.20
ROOT = Path(__file__).resolve().parents[1]


def read_lines(path: Path) -> list[str]:
    return [
        line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def frame_id(line: str) -> int:
    return int(float(line.split(",", 1)[0]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    dataset = yaml.safe_load(args.dataset_config.read_text(encoding="utf-8"))
    source_value = Path(dataset["data_root"])
    source_root = (
        source_value.resolve()
        if source_value.is_absolute()
        else (ROOT / source_value).resolve()
    )
    output_root = args.output_root.resolve()
    for sequence in dataset["sequences"]:
        source = source_root / sequence
        destination = output_root / sequence
        (destination / "gt").mkdir(parents=True, exist_ok=True)
        source_gt = source / dataset["reid_source_gt_file"]
        lines = read_lines(source_gt)
        frames = sorted({frame_id(line) for line in lines})
        cut = max(
            1, min(len(frames) - 1, int(round((1.0 - DEV_FRACTION) * len(frames))))
        )
        fit_frames = set(frames[:cut])
        dev_frames = set(frames[cut:])
        fit = [line for line in lines if frame_id(line) in fit_frames]
        dev = [line for line in lines if frame_id(line) in dev_frames]
        (destination / "gt" / "gt_train_fit.txt").write_text(
            "\n".join(fit) + "\n", encoding="utf-8"
        )
        (destination / "gt" / "gt_dev.txt").write_text(
            "\n".join(dev) + "\n", encoding="utf-8"
        )
        shutil.copy2(source / dataset["full_gt_file"], destination / "gt" / "gt.txt")
        image_link = destination / dataset["image_dir"]
        if not image_link.exists():
            image_link.symlink_to(
                source / dataset["image_dir"], target_is_directory=True
            )
        seqinfo = source / "seqinfo.ini"
        if seqinfo.is_file():
            shutil.copy2(seqinfo, destination / "seqinfo.ini")
        print(
            f"{sequence}: fit_frames={len(fit_frames)} dev_frames={len(dev_frames)} "
            f"fit_rows={len(fit)} dev_rows={len(dev)}"
        )


if __name__ == "__main__":
    main()
