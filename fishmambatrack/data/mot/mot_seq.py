"""Video-sequence reader: frames + detections/gt per frame.

fishmambatrack.data.mot.mot_seq

Sequence reader for MOT-style datasets (img1 + gt + det).
Supports split GT files that are re-indexed from 1 by inferring a frame offset
to map local frames -> global image frames.

Example sequence folder:
  BT-001/
    img1/000001.jpg ...
    gt/gt.txt
    gt/gt_train_half.txt
    gt/gt_val_half.txt
    det/det.txt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

from .mot_parser import MotRecord, index_by_frame, read_mot_file, summarize_records
from .mot_utils import (
    ImageIndex,
    build_image_index,
    infer_det_is_global,
    infer_frame_offset_from_full_gt,
)


@dataclass
class FrameData:
    seq_name: str
    frame: int  # local frame index (as in split gt)
    global_frame: int  # global frame index (image file index)
    img_path: Path
    det: List[MotRecord]
    gt: List[MotRecord]


def _remap_record_frame(r: MotRecord, new_frame: int) -> MotRecord:
    return MotRecord(
        frame=new_frame,
        track_id=r.track_id,
        tlwh=r.tlwh,
        score=r.score,
        extras=r.extras,
    )


class MOTSequence:
    """
    A view of one MOT sequence, optionally with split GT (train/val half).

    Internal convention:
      - `frame` is LOCAL frame id (same as what appears in the chosen GT file).
      - `global_frame = frame + frame_offset` is used to find image files and slice global det.

    If your split GT is re-indexed from 1 (common), `frame_offset` will be inferred
    by matching split frames to full GT frames.
    """

    def __init__(
        self,
        seq_dir: Union[str, Path],
        *,
        gt_relpath: str = "gt/gt.txt",
        det_relpath: Optional[str] = "det/det.txt",
        full_gt_relpath: str = "gt/gt.txt",
        img_dir_name: str = "img1",
        frame_offset: Optional[int] = None,
        det_is_global: Optional[bool] = None,
        drop_ignored_gt: bool = True,
    ) -> None:
        self.seq_dir = Path(seq_dir)
        if not self.seq_dir.exists():
            raise FileNotFoundError(self.seq_dir)

        self.seq_name = self.seq_dir.name
        self.img_dir = self.seq_dir / img_dir_name
        self.img_index: ImageIndex = build_image_index(self.img_dir)

        # Paths
        self.gt_path = self.seq_dir / gt_relpath
        if not self.gt_path.exists():
            raise FileNotFoundError(self.gt_path)

        self.full_gt_path = self.seq_dir / full_gt_relpath
        self.det_path = (self.seq_dir / det_relpath) if det_relpath else None
        if self.det_path is not None and (not self.det_path.exists()):
            # allow missing det if user wants GT-only
            self.det_path = None

        # Load GT (local frame space)
        gt_recs = read_mot_file(
            self.gt_path,
            is_gt=True,
            drop_ignored=drop_ignored_gt,
            strict=False,
            sort=True,
        )
        self.gt_by_frame: Dict[int, List[MotRecord]] = index_by_frame(gt_recs)
        gt_stats = summarize_records(gt_recs)
        self.gt_min_frame = int(gt_stats.get("min_frame", 1.0)) if gt_recs else 1
        self.gt_max_frame = int(gt_stats.get("max_frame", 0.0)) if gt_recs else 0

        # Decide / infer frame_offset
        if frame_offset is not None:
            self.frame_offset = int(frame_offset)
        else:
            # If using a split GT file (not equal to full gt) and full gt exists, infer offset
            if self.full_gt_path.exists() and (
                self.gt_path.resolve() != self.full_gt_path.resolve()
            ):
                self.frame_offset = infer_frame_offset_from_full_gt(
                    self.gt_path, self.full_gt_path
                )
            else:
                self.frame_offset = 0

        # Define local frame range we will iterate
        # Usually GT covers every frame; we use continuous range for safety.
        if self.gt_max_frame <= 0:
            raise RuntimeError(f"No GT records loaded from {self.gt_path}")
        self.local_frames: List[int] = list(
            range(self.gt_min_frame, self.gt_max_frame + 1)
        )

        # Load DET (convert to local frame space if det is global)
        self.det_by_frame: Dict[int, List[MotRecord]] = {}
        self.det_is_global = False
        if self.det_path is not None:
            det_recs = read_mot_file(
                self.det_path,
                is_gt=False,
                drop_ignored=False,
                strict=False,
                sort=True,
            )
            det_stats = summarize_records(det_recs)
            det_max = int(det_stats.get("max_frame", 0.0)) if det_recs else 0

            if det_is_global is not None:
                self.det_is_global = bool(det_is_global)
            else:
                self.det_is_global = infer_det_is_global(
                    det_max_frame=det_max,
                    gt_max_frame=self.gt_max_frame,
                    img_max_frame=self.img_index.max_frame,
                )

            det_by_global = index_by_frame(det_recs)
            if self.det_is_global:
                # global -> local using offset: local = global - offset
                for g_f, recs in det_by_global.items():
                    l_f = g_f - self.frame_offset
                    if l_f < self.gt_min_frame or l_f > self.gt_max_frame:
                        continue
                    self.det_by_frame[l_f] = [_remap_record_frame(r, l_f) for r in recs]
            else:
                # already local frame space
                self.det_by_frame = det_by_global

    def __len__(self) -> int:
        return len(self.local_frames)

    def local_to_global(self, frame: int) -> int:
        return frame + self.frame_offset

    def get_image_path(self, frame: int) -> Path:
        g = self.local_to_global(frame)
        if g not in self.img_index.frame_to_path:
            raise KeyError(
                f"[{self.seq_name}] No image for global frame {g}. "
                f"Local frame={frame}, offset={self.frame_offset}. "
                f"Image range=[{self.img_index.min_frame},{self.img_index.max_frame}]"
            )
        return self.img_index.frame_to_path[g]

    def get_frame(self, frame: int) -> FrameData:
        img_path = self.get_image_path(frame)
        det = self.det_by_frame.get(frame, [])
        gt = self.gt_by_frame.get(frame, [])
        return FrameData(
            seq_name=self.seq_name,
            frame=frame,
            global_frame=self.local_to_global(frame),
            img_path=img_path,
            det=det,
            gt=gt,
        )

    def iter_frames(self) -> Iterator[FrameData]:
        for f in self.local_frames:
            yield self.get_frame(f)

    def summary(self) -> Dict[str, object]:
        return {
            "seq_name": self.seq_name,
            "seq_dir": str(self.seq_dir),
            "img_dir": str(self.img_dir),
            "num_images": self.img_index.num_frames,
            "img_min_frame": self.img_index.min_frame,
            "img_max_frame": self.img_index.max_frame,
            "gt_path": str(self.gt_path),
            "gt_min_frame": self.gt_min_frame,
            "gt_max_frame": self.gt_max_frame,
            "frame_offset": self.frame_offset,
            "det_path": str(self.det_path) if self.det_path else None,
            "det_is_global": self.det_is_global,
        }


def _main() -> None:
    ap = argparse.ArgumentParser(
        "Inspect one MOT sequence view (with split-gt support)."
    )
    ap.add_argument(
        "--seq_dir",
        type=str,
        required=True,
        help="Sequence folder, e.g. data/MFT25-train/BT-001",
    )
    ap.add_argument(
        "--gt", type=str, default="gt/gt.txt", help="GT relative path inside sequence"
    )
    ap.add_argument(
        "--full_gt",
        type=str,
        default="gt/gt.txt",
        help="Full GT relative path (for offset inference)",
    )
    ap.add_argument(
        "--det",
        type=str,
        default="det/det.txt",
        help="DET relative path inside sequence",
    )
    ap.add_argument("--no_det", action="store_true", help="Do not load det")
    ap.add_argument(
        "--frame_offset",
        type=int,
        default=None,
        help="Manually set offset (override inference)",
    )
    args = ap.parse_args()

    det_rel = None if args.no_det else args.det

    seq = MOTSequence(
        args.seq_dir,
        gt_relpath=args.gt,
        det_relpath=det_rel,
        full_gt_relpath=args.full_gt,
        frame_offset=args.frame_offset,
    )

    s = seq.summary()
    print("Summary:", s)

    # Print a couple frames to validate mapping
    f0 = seq.local_frames[0]
    f1 = seq.local_frames[-1]
    d0 = seq.get_frame(f0)
    d1 = seq.get_frame(f1)
    print(
        f"\nFirst local frame={f0} -> global={d0.global_frame} "
        f"img={d0.img_path.name} gt={len(d0.gt)} det={len(d0.det)}"
    )
    print(
        f"Last  local frame={f1} -> global={d1.global_frame} "
        f"img={d1.img_path.name} gt={len(d1.gt)} det={len(d1.det)}"
    )

    # quick sanity: ensure all frames have an image
    missing = 0
    for f in (f0, (f0 + f1) // 2, f1):
        try:
            _ = seq.get_image_path(f)
        except Exception:
            missing += 1
    if missing == 0:
        print("\nImage mapping sanity check: OK (sampled frames exist).")
    else:
        print("\nImage mapping sanity check: FAILED (some sampled frames missing).")


if __name__ == "__main__":
    _main()
