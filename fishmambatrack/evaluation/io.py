"""
fishmambatrack.evaluation.io

Read/write MOT result files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple, Union


def write_mot_results(
    path: Union[str, Path],
    records: Iterable[Tuple[int, int, float, float, float, float, float]],
) -> None:
    """
    records: iterable of (frame, track_id, x, y, w, h, score)
    Write MOTChallenge format:
      frame,id,x,y,w,h,score,-1,-1,-1
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("w", encoding="utf-8") as f:
        for fr, tid, x, y, w, h, s in records:
            line = f"{int(fr)},{int(tid)},{float(x):.3f},{float(y):.3f},{float(w):.3f},{float(h):.3f},{float(s):.6f},-1,-1,-1\n"
            f.write(line)
