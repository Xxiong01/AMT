# fishmambatrack/data/mot/det_resolver.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple

# 默认优先用 YOLOX 离线 det
DET_PRIMARY_NAME = "det_yolox_ckpt.txt"
DET_FALLBACK_NAME = "det.txt"
DET_SUBDIR = "det"

def resolve_det_path(
    seq_dir: Path,
    primary: str = DET_PRIMARY_NAME,
    fallback: str = DET_FALLBACK_NAME,
    required_primary: bool = True,
) -> Path:
    """
    Return absolute det path for a sequence directory.

    required_primary=True: 必须存在 primary，否则报错（推荐，避免混用 det）
    required_primary=False: primary 不存在则回退 fallback（更宽松）
    """
    seq_dir = Path(seq_dir)
    p_primary = seq_dir / DET_SUBDIR / primary
    if p_primary.exists():
        return p_primary

    if required_primary:
        raise FileNotFoundError(
            f"[det_resolver] Required det not found: {p_primary}\n"
            f"  Hint: run export_det_from_checkpoint_yolox.py for this sequence."
        )

    p_fallback = seq_dir / DET_SUBDIR / fallback
    if p_fallback.exists():
        return p_fallback

    raise FileNotFoundError(
        f"[det_resolver] Neither det file exists:\n"
        f"  primary : {p_primary}\n"
        f"  fallback: {p_fallback}"
    )

def resolve_det_rel(
    seq_dir: Path,
    **kwargs
) -> str:
    """
    Return 'det/xxx.txt' style path (relative to seq_dir),
    useful when downstream expects a relative det path.
    """
    p = resolve_det_path(seq_dir, **kwargs)
    return f"{DET_SUBDIR}/{p.name}"
