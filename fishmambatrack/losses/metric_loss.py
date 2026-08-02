"""
Metric learning losses for ReID.

Includes:
- Batch-hard / batch-all triplet loss
- Contrastive loss
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_distance(
    embeddings: torch.Tensor,
    *,
    squared: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute pairwise Euclidean distance matrix.

    embeddings: (N,D)
    returns: (N,N)
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got {tuple(embeddings.shape)}")

    dot = embeddings @ embeddings.t()
    sq = torch.diagonal(dot)
    dist = sq.unsqueeze(1) - 2.0 * dot + sq.unsqueeze(0)
    dist = torch.clamp(dist, min=0.0)
    if not squared:
        dist = torch.sqrt(dist + eps)
    return dist


def cosine_distance(embeddings: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute pairwise cosine distance matrix = 1 - cosine_sim.
    embeddings: (N,D)
    returns: (N,N)
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got {tuple(embeddings.shape)}")
    x = F.normalize(embeddings, dim=1, eps=eps)
    sim = x @ x.t()
    return (1.0 - sim).clamp(min=0.0)


def _normalize_labels(labels: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    if labels.ndim != 1:
        labels = labels.view(-1)
    return labels.to(device=device, dtype=torch.long)


def _distance_matrix(
    embeddings: torch.Tensor,
    *,
    metric: str,
    normalize: bool,
    eps: float,
) -> torch.Tensor:
    if embeddings.dtype in (torch.float16, torch.bfloat16):
        embeddings = embeddings.float()
    if normalize:
        embeddings = F.normalize(embeddings, dim=1, eps=eps)
    metric = str(metric).lower()
    if metric == "cosine":
        return cosine_distance(embeddings, eps=eps)
    if metric == "euclidean":
        return pairwise_distance(embeddings, squared=False, eps=eps)
    if metric == "euclidean_squared":
        return pairwise_distance(embeddings, squared=True, eps=eps)
    raise ValueError(f"Unknown metric='{metric}' (expected euclidean/euclidean_squared/cosine).")


def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin: float = 0.3,
    metric: str = "euclidean",
    normalize: bool = False,
    eps: float = 1e-12,
    return_stats: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Batch-hard triplet loss (Hermans et al.).
    """
    device = embeddings.device
    labels = _normalize_labels(labels, device=device)
    dist = _distance_matrix(embeddings, metric=metric, normalize=normalize, eps=eps)

    N = dist.size(0)
    mask_pos = labels.unsqueeze(0) == labels.unsqueeze(1)
    mask_pos.fill_diagonal_(False)
    mask_neg = ~mask_pos

    dist_pos = dist.clone()
    dist_pos[~mask_pos] = -1.0
    hard_pos, _ = dist_pos.max(dim=1)

    dist_neg = dist.clone()
    dist_neg[~mask_neg] = torch.finfo(dist_neg.dtype).max
    hard_neg, _ = dist_neg.min(dim=1)

    valid = mask_pos.any(dim=1) & mask_neg.any(dim=1)
    if not valid.any():
        loss = dist.sum() * 0.0
        stats = {"valid_frac": 0.0, "pos_dist_mean": 0.0, "neg_dist_mean": 0.0}
        return (loss, stats) if return_stats else (loss, {})

    if margin is None or float(margin) <= 0.0:
        loss_vec = F.softplus(hard_pos - hard_neg)
    else:
        loss_vec = F.relu(hard_pos - hard_neg + float(margin))

    loss = loss_vec[valid].mean()

    stats = {
        "valid_frac": float(valid.float().mean().item()),
        "pos_dist_mean": float(hard_pos[valid].mean().item()),
        "neg_dist_mean": float(hard_neg[valid].mean().item()),
    }
    return (loss, stats) if return_stats else (loss, {})


def batch_all_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin: float = 0.3,
    metric: str = "euclidean",
    normalize: bool = False,
    eps: float = 1e-12,
    return_stats: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Batch-all triplet loss using all valid (a,p,n) triplets.
    """
    device = embeddings.device
    labels = _normalize_labels(labels, device=device)
    dist = _distance_matrix(embeddings, metric=metric, normalize=normalize, eps=eps)

    N = dist.size(0)
    mask_pos = labels.unsqueeze(0) == labels.unsqueeze(1)
    mask_pos.fill_diagonal_(False)
    mask_neg = ~mask_pos

    dist_ap = dist.unsqueeze(2)
    dist_an = dist.unsqueeze(1)
    triplet = dist_ap - dist_an
    if margin is None or float(margin) <= 0.0:
        triplet = F.softplus(triplet)
    else:
        triplet = F.relu(triplet + float(margin))

    mask = mask_pos.unsqueeze(2) & mask_neg.unsqueeze(1)
    if mask.sum() == 0:
        loss = dist.sum() * 0.0
        stats = {"valid_frac": 0.0, "num_triplets": 0.0}
        return (loss, stats) if return_stats else (loss, {})

    valid_triplet = triplet[mask]
    loss = valid_triplet.mean()
    stats = {
        "valid_frac": float((valid_triplet > 0.0).float().mean().item()),
        "num_triplets": float(valid_triplet.numel()),
    }
    return (loss, stats) if return_stats else (loss, {})


class TripletLoss(nn.Module):
    def __init__(
        self,
        margin: float = 0.3,
        *,
        metric: str = "euclidean",
        mining: str = "hard",
        normalize: bool = False,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.margin = margin
        self.metric = str(metric).lower()
        self.mining = str(mining).lower()
        self.normalize = bool(normalize)
        self.eps = float(eps)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor, *, return_stats: bool = False):
        if self.mining in ("hard", "batch_hard"):
            loss, stats = batch_hard_triplet_loss(
                embeddings,
                labels,
                margin=self.margin,
                metric=self.metric,
                normalize=self.normalize,
                eps=self.eps,
                return_stats=return_stats,
            )
        elif self.mining in ("all", "batch_all"):
            loss, stats = batch_all_triplet_loss(
                embeddings,
                labels,
                margin=self.margin,
                metric=self.metric,
                normalize=self.normalize,
                eps=self.eps,
                return_stats=return_stats,
            )
        else:
            raise ValueError(f"Unknown mining='{self.mining}' (expected hard/batch_hard/all/batch_all).")
        return (loss, stats) if return_stats else loss


class ContrastiveLoss(nn.Module):
    def __init__(
        self,
        margin: float = 0.5,
        *,
        metric: str = "euclidean",
        normalize: bool = False,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.margin = float(margin)
        self.metric = str(metric).lower()
        self.normalize = bool(normalize)
        self.eps = float(eps)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor, *, return_stats: bool = False):
        device = embeddings.device
        labels = _normalize_labels(labels, device=device)
        dist = _distance_matrix(embeddings, metric=self.metric, normalize=self.normalize, eps=self.eps)

        N = dist.size(0)
        mask_pos = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask_pos.fill_diagonal_(False)
        mask_neg = ~mask_pos

        dist2 = dist * dist
        pos_loss = dist2[mask_pos]
        neg_margin = F.relu(self.margin - dist)
        neg_loss = (neg_margin * neg_margin)[mask_neg]

        if pos_loss.numel() == 0 or neg_loss.numel() == 0:
            loss = dist.sum() * 0.0
            stats = {"pos_pairs": float(pos_loss.numel()), "neg_pairs": float(neg_loss.numel())}
            return (loss, stats) if return_stats else loss

        loss = 0.5 * (pos_loss.mean() + neg_loss.mean())
        stats = {"pos_pairs": float(pos_loss.numel()), "neg_pairs": float(neg_loss.numel())}
        return (loss, stats) if return_stats else loss
