#!/usr/bin/env python3
"""Train a declared temporal ReID encoder on the Train-derived Dev protocol."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_CONFIG = ROOT / "configs" / "models" / "mamba_l48.yaml"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fishmambatrack.data.mot.mot_reid_temporal import (  # noqa: E402
    MOTReIDTrackletDataset,
)
from fishmambatrack.data.samplers import RandomIdentitySampler  # noqa: E402
from fishmambatrack.losses.metric_loss import TripletLoss  # noqa: E402
from fishmambatrack.models.reid.registry import build_model  # noqa: E402


def deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def build_dataset(
    config: Dict[str, Any], gt_key: str, cache_key: str
) -> MOTReIDTrackletDataset:
    data = config["data"]
    sequence = config["sequence"]
    return MOTReIDTrackletDataset(
        _repo_path(data["root"]),
        seq_glob=data["sequence_glob"],
        gt_name=data[gt_key],
        full_gt_name=data["full_gt_file"],
        seq_len=int(sequence["length"]),
        frame_stride=int(sequence["frame_stride"]),
        seq_stride=int(sequence["seq_stride"]),
        max_frame_gap=int(sequence["max_gap"]),
        crop_pad_ratio=float(data["crop_pad"]),
        out_size=tuple(data["input_size"]),
        normalize=True,
        cache_path=_repo_path(data[cache_key]),
    )


def write_curves(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images, return_logits=True)["logits"]
            loss = F.cross_entropy(logits, labels)
            total_loss += float(loss) * labels.numel()
            total_correct += int((logits.argmax(1) == labels).sum())
            total += labels.numel()
    return total_loss / max(1, total), total_correct / max(1, total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = yaml.safe_load(args.model_config.read_text(encoding="utf-8"))
    deterministic(args.seed)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.model_config, output / "model_config.yaml")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available; pass --device cpu."
        )

    train_set = build_dataset(config, "train_gt_file", "train_index_cache")
    dev_set = build_dataset(config, "dev_gt_file", "dev_index_cache")
    dev_indices = []
    for index, item in enumerate(dev_set.items):
        if item.track_key not in train_set.pid_map:
            continue
        item.pid = int(train_set.pid_map[item.track_key])
        dev_indices.append(index)
    if not dev_indices:
        raise RuntimeError(
            "Train-derived Dev contains no identities shared with the fit subset."
        )
    dev_data = Subset(dev_set, dev_indices)

    training = config["training"]
    batch_size = int(training["batch_size"])
    if int(training["epochs"]) <= 0:
        raise ValueError("training.epochs must be positive.")
    if batch_size <= 0 or int(training["evaluation_batch_size"]) <= 0:
        raise ValueError("Training and evaluation batch sizes must be positive.")
    if int(training["workers"]) < 0:
        raise ValueError("training.workers cannot be negative.")
    sampler = RandomIdentitySampler(
        train_set,
        num_instances=int(training["instances_per_identity"]),
        batch_size=batch_size,
        seed=args.seed,
    )
    if len(sampler) == 0:
        raise RuntimeError(
            "The training split cannot form one PK batch; reduce the batch size "
            "or instances_per_identity, or check the prepared split."
        )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=int(training["workers"]),
        pin_memory=bool(training["pin_memory"]),
        drop_last=True,
    )
    dev_loader = DataLoader(
        dev_data,
        batch_size=int(training["evaluation_batch_size"]),
        shuffle=False,
        num_workers=int(training["workers"]),
        pin_memory=bool(training["pin_memory"]),
    )

    model_values = dict(config["model"]["config"])
    model_name = str(config["model"]["name"])
    model = build_model(
        model_values,
        num_classes=train_set.num_pids,
        model_name=model_name,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    triplet = TripletLoss(margin=float(training["triplet_margin"]))
    use_mixed_precision = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_mixed_precision)
    curves: list[dict] = []
    best_loss = float("inf")

    for epoch in range(1, int(training["epochs"]) + 1):
        sampler.set_epoch(epoch)
        model.train()
        total_loss, total = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_mixed_precision):
                result = model(images, return_logits=True)
                cross_entropy = F.cross_entropy(result["logits"], labels)
                batch_hard_triplet = triplet(result["emb"], labels)
                loss = cross_entropy + batch_hard_triplet
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * labels.numel()
            total += labels.numel()

        dev_loss, dev_accuracy = evaluate(model, dev_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, total),
            "dev_ce": dev_loss,
            "dev_accuracy": dev_accuracy,
        }
        curves.append(row)
        write_curves(output / "learning_curves.csv", curves)
        if dev_loss < best_loss:
            best_loss = dev_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "meta": {
                        "model_name": model_name,
                        "model_cfg": model_values,
                        "sequence_length": int(config["sequence"]["length"]),
                        "pool_mode": str(config["model"].get("pool_mode", "mean_last")),
                        "embedding_dimension": int(model_values.get("emb_dim", 256)),
                        "training_seed": int(args.seed),
                        "selection_split": "MFT25-Train-derived Dev",
                        "final_val_used_for_selection": False,
                        "input_size": list(config["data"]["input_size"]),
                        "crop_padding": float(config["data"]["crop_pad"]),
                    },
                },
                output / "best.pt",
            )
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
