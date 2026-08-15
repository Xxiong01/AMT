"""
Samplers: identity-based sampling for metric learning.

Common pattern: PK sampling (P identities, K instances each).
"""

from __future__ import annotations

import random
from typing import Dict, Iterator, List

from torch.utils.data import Sampler


def _build_pid_to_indices(data_source) -> Dict[int, List[int]]:
    if hasattr(data_source, "dataset") and hasattr(data_source, "indices"):
        base = data_source.dataset
        indices = list(data_source.indices)
        base_to_sub = {int(bi): int(si) for si, bi in enumerate(indices)}
        pid_to_indices: Dict[int, List[int]] = {}

        if hasattr(base, "items"):
            for sub_i, base_i in enumerate(indices):
                pid = int(getattr(base.items[int(base_i)], "pid"))
                pid_to_indices.setdefault(pid, []).append(int(sub_i))
            if pid_to_indices:
                return pid_to_indices

        base_pid_to_indices = _build_pid_to_indices(base)
        for pid, base_list in base_pid_to_indices.items():
            sub_list = [base_to_sub[i] for i in base_list if i in base_to_sub]
            if sub_list:
                pid_to_indices[int(pid)] = sub_list
        if pid_to_indices:
            return pid_to_indices

    if hasattr(data_source, "pid_to_indices"):
        pid_to_indices = data_source.pid_to_indices
        return {int(k): list(v) for k, v in pid_to_indices.items()}

    if hasattr(data_source, "items"):
        pid_to_indices: Dict[int, List[int]] = {}
        for i, it in enumerate(data_source.items):
            pid = int(getattr(it, "pid"))
            pid_to_indices.setdefault(pid, []).append(i)
        if pid_to_indices:
            return pid_to_indices

    raise ValueError("data_source must expose pid_to_indices or items with .pid.")


def _build_pk_batches(
    pid_to_indices: Dict[int, List[int]],
    *,
    num_instances: int,
    num_pids_per_batch: int,
    rng: random.Random,
    drop_last: bool,
) -> List[List[int]]:
    pid_to_groups: Dict[int, List[List[int]]] = {}
    for pid, idxs in pid_to_indices.items():
        idxs = list(idxs)
        if len(idxs) == 0:
            continue
        if len(idxs) < num_instances:
            idxs = idxs + rng.choices(idxs, k=num_instances - len(idxs))
        else:
            rng.shuffle(idxs)

        groups = [
            idxs[i : i + num_instances] for i in range(0, len(idxs), num_instances)
        ]
        if groups and len(groups[-1]) < num_instances:
            pad = rng.choices(idxs, k=num_instances - len(groups[-1]))
            groups[-1] = groups[-1] + pad
        pid_to_groups[int(pid)] = groups

    avail_pids = [pid for pid, groups in pid_to_groups.items() if groups]
    batches: List[List[int]] = []
    while len(avail_pids) >= num_pids_per_batch:
        chosen = rng.sample(avail_pids, num_pids_per_batch)
        batch: List[int] = []
        for pid in chosen:
            batch.extend(pid_to_groups[pid].pop(0))
            if not pid_to_groups[pid]:
                avail_pids.remove(pid)
        batches.append(batch)

    if (not drop_last) and avail_pids:
        batch = []
        while len(batch) < num_pids_per_batch * num_instances:
            pid = rng.choice(avail_pids)
            if pid_to_groups[pid]:
                batch.extend(pid_to_groups[pid].pop(0))
            else:
                idxs = pid_to_indices[pid]
                batch.extend(rng.choices(idxs, k=num_instances))
        batches.append(batch[: num_pids_per_batch * num_instances])

    return batches


class RandomIdentitySampler(Sampler[int]):
    """
    PK sampler that returns a flat index stream (DataLoader groups into batches).
    """

    def __init__(
        self,
        data_source,
        *,
        num_instances: int = 4,
        batch_size: int = 64,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        if int(num_instances) <= 0 or int(batch_size) <= 0:
            raise ValueError("num_instances and batch_size must be positive.")
        if int(batch_size) % int(num_instances) != 0:
            raise ValueError("batch_size must be divisible by num_instances.")

        self.pid_to_indices = _build_pid_to_indices(data_source)
        self.num_instances = int(num_instances)
        self.batch_size = int(batch_size)
        self.num_pids_per_batch = int(batch_size // num_instances)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        batches = _build_pk_batches(
            self.pid_to_indices,
            num_instances=self.num_instances,
            num_pids_per_batch=self.num_pids_per_batch,
            rng=random.Random(self.seed + self.epoch),
            drop_last=self.drop_last,
        )
        return len(batches) * self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        batches = _build_pk_batches(
            self.pid_to_indices,
            num_instances=self.num_instances,
            num_pids_per_batch=self.num_pids_per_batch,
            rng=rng,
            drop_last=self.drop_last,
        )
        flat = [i for b in batches for i in b]
        return iter(flat)


class RandomIdentityBatchSampler(Sampler[List[int]]):
    """
    PK sampler that yields lists of indices (batch sampler).
    """

    def __init__(
        self,
        data_source,
        *,
        num_instances: int = 4,
        batch_size: int = 64,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        if int(num_instances) <= 0 or int(batch_size) <= 0:
            raise ValueError("num_instances and batch_size must be positive.")
        if int(batch_size) % int(num_instances) != 0:
            raise ValueError("batch_size must be divisible by num_instances.")

        self.pid_to_indices = _build_pid_to_indices(data_source)
        self.num_instances = int(num_instances)
        self.batch_size = int(batch_size)
        self.num_pids_per_batch = int(batch_size // num_instances)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(
            _build_pk_batches(
                self.pid_to_indices,
                num_instances=self.num_instances,
                num_pids_per_batch=self.num_pids_per_batch,
                rng=random.Random(self.seed + self.epoch),
                drop_last=self.drop_last,
            )
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        batches = _build_pk_batches(
            self.pid_to_indices,
            num_instances=self.num_instances,
            num_pids_per_batch=self.num_pids_per_batch,
            rng=rng,
            drop_last=self.drop_last,
        )
        return iter(batches)
