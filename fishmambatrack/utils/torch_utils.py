"""Torch runtime helpers."""

from __future__ import annotations


def enable_tf32() -> bool:
    try:
        import torch
    except Exception:
        return False

    if not torch.cuda.is_available():
        return False

    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass

    try:
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    return True
