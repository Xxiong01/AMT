"""Deterministic image cropping and normalization for AMT-L48."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
from PIL import Image


def crop_tlwh(
    image: Image.Image,
    tlwh: Tuple[float, float, float, float],
    *,
    pad_ratio: float = 0.10,
) -> Image.Image:
    width, height = image.size
    x, y, box_width, box_height = map(float, tlwh)
    pad_x, pad_y = box_width * pad_ratio, box_height * pad_ratio
    x1 = max(0, min(width - 1, int(math.floor(x - pad_x))))
    y1 = max(0, min(height - 1, int(math.floor(y - pad_y))))
    x2 = max(1, min(width, int(math.ceil(x + box_width + pad_x))))
    y2 = max(1, min(height, int(math.ceil(y + box_height + pad_y))))
    return image.crop((x1, y1, max(x1 + 2, x2), max(y1 + 2, y2)))


def default_transform(
    image: Image.Image,
    *,
    size: Tuple[int, int] = (256, 128),
    normalize: bool = True,
) -> torch.Tensor:
    image = image.convert("RGB").resize((size[1], size[0]), resample=Image.BILINEAR)
    value = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    tensor = torch.from_numpy(value)
    if normalize:
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        tensor = (tensor - mean) / std
    return tensor
