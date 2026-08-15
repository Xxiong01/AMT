"""Shared AMT-L48 ReID backbone helper."""

from __future__ import annotations

import torch
import torch.nn as nn


def forward_backbone(
    model: nn.Module, backbone: nn.Module, images: torch.Tensor
) -> torch.Tensor:
    del model
    return backbone(images)
