"""Batch-hard triplet loss used by the AMT-L48 training objective."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_distance(embeddings: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional embedding tensor, got {embeddings.shape}."
        )
    if embeddings.dtype in (torch.float16, torch.bfloat16):
        embeddings = embeddings.float()
    products = embeddings @ embeddings.t()
    squared_norms = torch.diagonal(products)
    squared = (
        squared_norms.unsqueeze(1) - 2.0 * products + squared_norms.unsqueeze(0)
    ).clamp(min=0.0)
    return torch.sqrt(squared + eps)


class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.margin = float(margin)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.view(-1).to(device=embeddings.device, dtype=torch.long)
        distances = pairwise_distance(embeddings)
        same_identity = labels.unsqueeze(0) == labels.unsqueeze(1)
        same_identity.fill_diagonal_(False)
        different_identity = ~same_identity
        different_identity.fill_diagonal_(False)

        positive_distances = distances.masked_fill(~same_identity, -1.0)
        negative_distances = distances.masked_fill(
            ~different_identity, torch.finfo(distances.dtype).max
        )
        hardest_positive = positive_distances.max(dim=1).values
        hardest_negative = negative_distances.min(dim=1).values
        valid = same_identity.any(dim=1) & different_identity.any(dim=1)
        if not valid.any():
            return distances.sum() * 0.0
        return F.relu(
            hardest_positive[valid] - hardest_negative[valid] + self.margin
        ).mean()
