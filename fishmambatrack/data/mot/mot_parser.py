"""MOT format parser for gt/det files.

fishmambatrack.data.mot.mot_parser

Expected line format (MOTChallenge):
frame, id, x, y, w, h, conf, class, visibility

We will be robust to fewer columns.

Robust parser for MOT-style annotation files. We target the common MOTChallenge CSV format:

    frame, id, bb_left, bb_top, bb_width, bb_height, conf, ...

- Ground truth (gt/gt*.txt) usually has `conf` in column 7:
    * conf==1 : valid object
    * conf==0 : ignored region / do-not-care

- Detections (det/det.txt) usually has `id` == -1 and `conf` is the detection score.

This parser is intentionally tolerant:
- Supports both comma-separated and whitespace-separated lines.
- Ignores blank lines and comment lines starting with '#'.
- Can skip non-numeric header rows.

The rest of the pipeline only needs (frame, id, x, y, w, h, score).
Any additional columns are stored in `extras`.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union


class MotFormatError(ValueError):
    """Raised when a MOT file line cannot be parsed."""


@dataclass(frozen=True)
class MotRecord:
    """One row in a MOT gt/det file."""

    frame: int
    track_id: int
    tlwh: Tuple[float, float, float, float]  # (x, y, w, h)
    score: float = 1.0
    extras: Tuple[float, ...] = ()

    @property
    def x(self) -> float:
        return self.tlwh[0]

    @property
    def y(self) -> float:
        return self.tlwh[1]

    @property
    def w(self) -> float:
        return self.tlwh[2]

    @property
    def h(self) -> float:
        return self.tlwh[3]

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        x, y, w, h = self.tlwh
        return (x, y, x + w, y + h)


_SEP_RE = re.compile(r"[\s,]+")


def _is_number_token(tok: str) -> bool:
    """Heuristic check for header rows."""
    try:
        float(tok)
        return True
    except Exception:
        return False


def _split_tokens(line: str) -> List[str]:
    """Split a MOT line into tokens (comma/space tolerant)."""
    # strip inline comments
    if "#" in line:
        line = line.split("#", 1)[0]
    line = line.strip()
    if not line:
        return []
    return [t for t in _SEP_RE.split(line) if t]


def _parse_float(tok: str, *, path: Path, line_no: int, strict: bool) -> Optional[float]:
    try:
        return float(tok)
    except Exception:
        if strict:
            raise MotFormatError(f"Cannot parse float token='{tok}' at {path}:{line_no}")
        return None


def _parse_int(tok: str, *, path: Path, line_no: int, strict: bool) -> Optional[int]:
    try:
        # Some tools write '1.0' for ints
        return int(float(tok))
    except Exception:
        if strict:
            raise MotFormatError(f"Cannot parse int token='{tok}' at {path}:{line_no}")
        return None


def parse_mot_line(
    tokens: Sequence[str],
    *,
    path: Path,
    line_no: int,
    strict: bool = False,
    default_score: float = 1.0,
) -> Optional[MotRecord]:
    """Parse a tokenized MOT line.

    Returns None if the line should be skipped (e.g., header) and strict=False.
    """
    if len(tokens) == 0:
        return None

    # Skip header-like rows when not strict
    if not _is_number_token(tokens[0]):
        if strict:
            raise MotFormatError(f"Non-numeric header token='{tokens[0]}' at {path}:{line_no}")
        return None

    if len(tokens) < 6:
        if strict:
            raise MotFormatError(
                f"Too few columns ({len(tokens)}) at {path}:{line_no}. Need >=6: frame,id,x,y,w,h"
            )
        return None

    frame = _parse_int(tokens[0], path=path, line_no=line_no, strict=strict)
    track_id = _parse_int(tokens[1], path=path, line_no=line_no, strict=strict)
    if frame is None or track_id is None:
        return None

    x = _parse_float(tokens[2], path=path, line_no=line_no, strict=strict)
    y = _parse_float(tokens[3], path=path, line_no=line_no, strict=strict)
    w = _parse_float(tokens[4], path=path, line_no=line_no, strict=strict)
    h = _parse_float(tokens[5], path=path, line_no=line_no, strict=strict)
    if x is None or y is None or w is None or h is None:
        return None

    score = default_score
    if len(tokens) >= 7:
        s = _parse_float(tokens[6], path=path, line_no=line_no, strict=strict)
        if s is None:
            return None
        score = float(s)

    extras_list: List[float] = []
    if len(tokens) > 7:
        for tok in tokens[7:]:
            v = _parse_float(tok, path=path, line_no=line_no, strict=strict)
            if v is None:
                # In non-strict mode, stop collecting extras once we hit non-numeric stuff.
                break
            extras_list.append(float(v))

    return MotRecord(
        frame=frame,
        track_id=track_id,
        tlwh=(float(x), float(y), float(w), float(h)),
        score=score,
        extras=tuple(extras_list),
    )


def iter_mot_records(
    path: Union[str, Path],
    *,
    is_gt: Optional[bool] = None,
    drop_ignored: bool = True,
    drop_invalid_bbox: bool = True,
    strict: bool = False,
    min_frame: Optional[int] = None,
    max_frame: Optional[int] = None,
) -> Iterator[MotRecord]:
    """Yield MotRecord from a MOT-style file.

    Args:
        path: Path to txt.
        is_gt: If True, apply GT-specific filtering (`drop_ignored`).
               If False, treat as detection file.
               If None, infer from path name containing '/gt/' or file name starting with 'gt'.
        drop_ignored: For GT, drop rows with score==0 (do-not-care).
        drop_invalid_bbox: Drop rows where w<=0 or h<=0.
        strict: If True, raise on parse errors; else skip bad lines.
        min_frame/max_frame: Optional frame range filter (inclusive).
    """
    p = Path(path)
    if is_gt is None:
        s = str(p).lower().replace("\\", "/")
        is_gt = ("/gt/" in s) or p.name.lower().startswith("gt")

    if not p.exists():
        raise FileNotFoundError(p)

    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            tokens = _split_tokens(line)
            if not tokens:
                continue

            rec = parse_mot_line(tokens, path=p, line_no=line_no, strict=strict)
            if rec is None:
                continue

            if drop_invalid_bbox and (rec.w <= 0.0 or rec.h <= 0.0):
                continue

            if min_frame is not None and rec.frame < min_frame:
                continue
            if max_frame is not None and rec.frame > max_frame:
                continue

            if is_gt and drop_ignored and rec.score <= 0.0:
                continue

            yield rec


def read_mot_file(
    path: Union[str, Path],
    *,
    is_gt: Optional[bool] = None,
    drop_ignored: bool = True,
    drop_invalid_bbox: bool = True,
    strict: bool = False,
    sort: bool = True,
    min_frame: Optional[int] = None,
    max_frame: Optional[int] = None,
) -> List[MotRecord]:
    """Read a MOT-style file into a list."""
    records = list(
        iter_mot_records(
            path,
            is_gt=is_gt,
            drop_ignored=drop_ignored,
            drop_invalid_bbox=drop_invalid_bbox,
            strict=strict,
            min_frame=min_frame,
            max_frame=max_frame,
        )
    )
    if sort:
        records.sort(key=lambda r: (r.frame, r.track_id))
    return records


def index_by_frame(records: Iterable[MotRecord]) -> Dict[int, List[MotRecord]]:
    """Group records by frame index."""
    out: Dict[int, List[MotRecord]] = {}
    for r in records:
        out.setdefault(r.frame, []).append(r)
    return out


def summarize_records(records: Sequence[MotRecord]) -> Dict[str, float]:
    """Basic stats for quick sanity checks."""
    if not records:
        return {"num_records": 0, "num_frames": 0, "num_ids": 0}

    frames = [r.frame for r in records]
    ids = [r.track_id for r in records]
    return {
        "num_records": float(len(records)),
        "num_frames": float(len(set(frames))),
        "num_ids": float(len(set(ids))),
        "min_frame": float(min(frames)),
        "max_frame": float(max(frames)),
    }


def _main() -> None:
    ap = argparse.ArgumentParser(description="Quickly inspect a MOT txt file.")
    ap.add_argument("path", type=str, help="Path to gt/det txt")
    ap.add_argument("--gt", action="store_true", help="Treat as GT file (drop conf==0 by default)")
    ap.add_argument("--det", action="store_true", help="Treat as DET file")
    ap.add_argument("--keep-ignored", action="store_true", help="Keep GT conf==0 rows")
    ap.add_argument("--strict", action="store_true", help="Strict parsing")
    ap.add_argument("--head", type=int, default=5, help="Print first N records")
    args = ap.parse_args()

    if args.gt and args.det:
        raise SystemExit("Choose only one of --gt or --det")

    is_gt = None
    if args.gt:
        is_gt = True
    if args.det:
        is_gt = False

    records = read_mot_file(
        args.path,
        is_gt=is_gt,
        drop_ignored=not args.keep_ignored,
        strict=args.strict,
        sort=True,
    )
    stats = summarize_records(records)
    print(f"Loaded: {args.path}")
    print("Stats:", stats)
    for r in records[: args.head]:
        print(r)


if __name__ == "__main__":
    _main()
