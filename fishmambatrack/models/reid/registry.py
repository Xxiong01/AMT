"""Construction and checkpoint loading for paper ReID encoders."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from .fishmamba_reid_temporal import FishMambaReIDTemporal, FishMambaReIDTemporalConfig
from .temporal_baselines import TemporalBaselineConfig, TemporalBaselineReID


def build_model(
    cfg: Dict[str, Any], num_classes: int = 0, *, model_name: str = "mamba"
) -> torch.nn.Module:
    values = dict(cfg)
    values["num_classes"] = int(num_classes)
    if model_name == "mamba":
        allowed = {
            field.name for field in dataclasses.fields(FishMambaReIDTemporalConfig)
        }
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"Unknown Mamba model configuration keys: {unknown}")
        return FishMambaReIDTemporal(FishMambaReIDTemporalConfig(**values))
    values["name"] = model_name
    allowed = {field.name for field in dataclasses.fields(TemporalBaselineConfig)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown temporal-baseline configuration keys: {unknown}")
    return TemporalBaselineReID(TemporalBaselineConfig(**values))


def _state_dict(payload: Any) -> Dict[str, torch.Tensor]:
    if isinstance(payload, dict) and isinstance(payload.get("model"), dict):
        return payload["model"]
    raise ValueError("best.pt does not contain the AMT-L48 model state.")


def load_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    cfg = dict(meta.get("model_cfg", {}))
    if not cfg:
        raise ValueError("best.pt is missing its AMT-L48 model configuration.")
    model_name = str(meta.get("model_name", "mamba"))
    model = build_model(cfg, num_classes=0, model_name=model_name).to(device)
    missing, unexpected = model.load_state_dict(_state_dict(payload), strict=False)
    unexpected = [key for key in unexpected if not key.startswith("classifier.")]
    missing = [key for key in missing if not key.startswith("classifier.")]
    if unexpected or missing:
        raise RuntimeError(
            f"Checkpoint/model mismatch; missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    model.eval()
    return model, dict(meta)
