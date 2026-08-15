"""Utilities: discover sequences, map frame->path, compute per-seq stats.

fishmambatrack.data.mot.mot_utils

Utilities for MOT-style datasets:
- Discover sequences under a dataset root
- Build image frame index from img1/
- Infer frame offset for split gt files (e.g., gt_train_half.txt / gt_val_half.txt)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .mot_parser import MotRecord, index_by_frame, read_mot_file


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_DIGITS_RE = re.compile(r"(\d+)")


@dataclass(frozen=True)
class ImageIndex:
    """Mapping from global frame id -> image path."""

    frame_to_path: Dict[int, Path]
    frames: List[int]
    img_dir: Path

    @property
    def min_frame(self) -> int:
        return self.frames[0] if self.frames else 0

    @property
    def max_frame(self) -> int:
        return self.frames[-1] if self.frames else 0

    @property
    def num_frames(self) -> int:
        return len(self.frames)


def _extract_frame_id(p: Path) -> Optional[int]:
    """
    Extract frame id from an image filename.
    Supports '000001.jpg', '1.png', 'img000123.jpg', etc.
    Uses the last digit group in the stem.
    """
    stem = p.stem
    ms = _DIGITS_RE.findall(stem)
    if not ms:
        return None
    return int(ms[-1])


def build_image_index(img_dir: Union[str, Path]) -> ImageIndex:
    img_dir = Path(img_dir)
    if not img_dir.exists():
        raise FileNotFoundError(img_dir)
    if not img_dir.is_dir():
        raise NotADirectoryError(img_dir)

    frame_to_path: Dict[int, Path] = {}
    for p in sorted(img_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in _IMG_EXTS:
            continue
        fid = _extract_frame_id(p)
        if fid is None:
            continue
        # keep first occurrence if duplicates
        frame_to_path.setdefault(fid, p)

    frames = sorted(frame_to_path.keys())
    if not frames:
        raise RuntimeError(
            f"No images found in {img_dir} with extensions={sorted(_IMG_EXTS)}"
        )
    return ImageIndex(frame_to_path=frame_to_path, frames=frames, img_dir=img_dir)


def is_mot_sequence_dir(seq_dir: Union[str, Path], img_dir_name: str = "img1") -> bool:
    seq_dir = Path(seq_dir)
    return seq_dir.is_dir() and (seq_dir / img_dir_name).is_dir()


def discover_sequence_dirs(
    root: Union[str, Path],
    *,
    seq_glob: str = "*",
    img_dir_name: str = "img1",
    sort: bool = True,
) -> List[Path]:
    """
    Discover sequence folders under dataset root.

    A valid sequence folder contains `img1/`.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    if not root.is_dir():
        raise NotADirectoryError(root)

    seq_dirs: List[Path] = []
    for d in root.glob(seq_glob):
        if is_mot_sequence_dir(d, img_dir_name=img_dir_name):
            seq_dirs.append(d)

    if sort:
        seq_dirs = sorted(seq_dirs, key=lambda p: p.name)
    return seq_dirs


def frame_signature(
    records: Sequence[MotRecord],
) -> Tuple[Tuple[int, int, int, int, int], ...]:
    """
    Build an exact signature for a frame based on (id, x, y, w, h) rounded to int.
    This is used to match split-GT frames back to full-GT frames and infer offsets.
    """
    sig = []
    for r in records:
        sig.append(
            (
                r.track_id,
                int(round(r.x)),
                int(round(r.y)),
                int(round(r.w)),
                int(round(r.h)),
            )
        )
    sig.sort()
    return tuple(sig)


def infer_frame_offset_from_full_gt(
    split_gt_path: Union[str, Path],
    full_gt_path: Union[str, Path],
    *,
    k_frames: int = 3,
) -> int:
    """
    Infer an offset such that:
        full_frame = split_frame + offset

    Works best when split GT is a contiguous subsequence of full GT
    whose frames are re-indexed from 1..T (common for half-splits).

    If cannot infer, returns 0.
    """
    split_gt_path = Path(split_gt_path)
    full_gt_path = Path(full_gt_path)
    if not split_gt_path.exists():
        raise FileNotFoundError(split_gt_path)
    if not full_gt_path.exists():
        # No full GT available -> cannot infer reliably
        return 0

    split_recs = read_mot_file(
        split_gt_path, is_gt=True, drop_ignored=True, strict=False, sort=True
    )
    full_recs = read_mot_file(
        full_gt_path, is_gt=True, drop_ignored=True, strict=False, sort=True
    )
    split_by_f = index_by_frame(split_recs)
    full_by_f = index_by_frame(full_recs)

    split_frames = sorted(split_by_f.keys())
    full_frames = sorted(full_by_f.keys())
    if not split_frames or not full_frames:
        return 0

    # Take first K split frames that exist
    split_frames_k = split_frames[: max(1, min(k_frames, len(split_frames)))]
    split_sigs = {sf: frame_signature(split_by_f[sf]) for sf in split_frames_k}

    # Precompute full frame signatures once
    full_sig_to_frames: Dict[Tuple[Tuple[int, int, int, int, int], ...], List[int]] = {}
    for ff in full_frames:
        sig = frame_signature(full_by_f[ff])
        full_sig_to_frames.setdefault(sig, []).append(ff)

    # Candidate offsets are frames where split first signature matches
    sf0 = split_frames_k[0]
    sig0 = split_sigs[sf0]
    candidate_full_frames = full_sig_to_frames.get(sig0, [])

    for ff0 in candidate_full_frames:
        offset = ff0 - sf0
        ok = True
        for sf in split_frames_k[1:]:
            ff = sf + offset
            if ff not in full_by_f:
                ok = False
                break
            if frame_signature(full_by_f[ff]) != split_sigs[sf]:
                ok = False
                break
        if ok:
            return offset

    # Fallback: if split already uses global indexing, offset should be 0
    return 0


def infer_det_is_global(
    det_max_frame: int,
    gt_max_frame: int,
    img_max_frame: int,
) -> bool:
    """
    Heuristic: if det spans the full image range (e.g. 3000),
    but gt split is shorter (e.g. 1501), det is global.
    """
    if det_max_frame == img_max_frame and gt_max_frame != img_max_frame:
        return True
    if det_max_frame > gt_max_frame:
        return True
    return False
