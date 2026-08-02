"""
fishmambatrack.models.backbones.mamba_blocks

Minimal Mamba encoder wrappers:
- MambaBlock: LN -> Mamba -> residual
- MambaEncoder: stack of blocks

Input/Output shape: (B, L, D)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


def _import_mamba():
    # Try multiple import paths for compatibility across versions.
    try:
        from mamba_ssm.modules.mamba_simple import Mamba  # type: ignore
        return Mamba
    except Exception:
        try:
            from mamba_ssm import Mamba  # type: ignore
            return Mamba
        except Exception as e:
            raise ImportError(
                "Cannot import Mamba from mamba_ssm. "
                "Make sure `mamba-ssm` is installed and importable."
            ) from e


Mamba = _import_mamba()


@dataclass
class MambaEncoderConfig:
    d_model: int = 256
    n_layers: int = 2
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.0


class MambaBlock(nn.Module):
    def __init__(self, cfg: MambaEncoderConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(cfg.d_model)
        self.mamba = Mamba(
            d_model=cfg.d_model,
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            expand=cfg.expand,
        )
        self.drop = nn.Dropout(p=float(cfg.dropout)) if cfg.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        y = self.mamba(self.norm(x))
        y = self.drop(y)
        return x + y


class MambaEncoder(nn.Module):
    def __init__(self, cfg: Optional[MambaEncoderConfig] = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = MambaEncoderConfig()
        self.cfg = cfg
        self.blocks = nn.ModuleList([MambaBlock(cfg) for _ in range(int(cfg.n_layers))])
        self.out_norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.out_norm(x)
