"""
fishmambatrack.models.reid.registry

Lazy registry + checkpoint loader for multiple ReID backbones.

Why:
  - FishMamba imports `mamba_ssm` and may not be available in every env.
  - We want to compare multiple ReID variants with the same tracker pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch


@dataclass
class ReIDCheckpointMeta:
    model_name: str
    model_cfg: Dict[str, Any]


def _extract_state_dict(ckpt_obj: Any) -> Dict[str, Any]:
    if isinstance(ckpt_obj, dict):
        for k in ("model", "state_dict", "net", "network"):
            v = ckpt_obj.get(k, None)
            if isinstance(v, dict):
                return v
        # maybe already a state_dict
        if all(isinstance(k, str) for k in ckpt_obj.keys()):
            return ckpt_obj  # type: ignore[return-value]
    raise ValueError("Unsupported checkpoint format (expect dict with a state_dict).")


def _extract_meta(ckpt_obj: Any) -> Optional[ReIDCheckpointMeta]:
    if not isinstance(ckpt_obj, dict):
        return None
    meta = ckpt_obj.get("meta", None)
    if not isinstance(meta, dict):
        return None
    model_name = meta.get("model_name", None)
    if not isinstance(model_name, str) or not model_name:
        return None
    model_cfg = meta.get("model_cfg", {}) if isinstance(meta.get("model_cfg", {}), dict) else {}
    return ReIDCheckpointMeta(model_name=model_name, model_cfg=dict(model_cfg))


def build_reid_model(
    model_name: str,
    *,
    model_cfg: Optional[Dict[str, Any]] = None,
    num_classes: Optional[int] = None,
) -> torch.nn.Module:
    name = str(model_name).lower()
    cfg_dict = {} if model_cfg is None else dict(model_cfg)

    if name in ("fishmamba", "mamba", "fishmamba_reid"):
        from fishmambatrack.models.reid.fishmamba_reid import FishMambaReID, FishMambaReIDConfig

        cfg = FishMambaReIDConfig(**cfg_dict)
        if num_classes is not None:
            cfg.num_classes = int(num_classes)
        return FishMambaReID(cfg)

    if name in ("fishmamba_temporal", "mamba_temporal", "fishmamba_seq", "temporal_mamba"):
        from fishmambatrack.models.reid.fishmamba_reid_temporal import (
            FishMambaReIDTemporal,
            FishMambaReIDTemporalConfig,
        )

        cfg = FishMambaReIDTemporalConfig(**cfg_dict)
        if num_classes is not None:
            cfg.num_classes = int(num_classes)
        return FishMambaReIDTemporal(cfg)

    if name in (
        "temporal_mean_pool",
        "temporal_gru",
        "temporal_lstm",
        "temporal_transformer_lite",
        "temporal_transformer",
        "temporal_baseline",
    ):
        from fishmambatrack.models.reid.temporal_baseline_reid import (
            FishTemporalBaselineReID,
            FishTemporalBaselineReIDConfig,
        )

        cfg_overrides: Dict[str, Any] = {}
        if name == "temporal_mean_pool":
            cfg_overrides["temporal_type"] = "mean_pool"
        elif name == "temporal_gru":
            cfg_overrides["temporal_type"] = "gru"
        elif name == "temporal_lstm":
            cfg_overrides["temporal_type"] = "lstm"
        elif name in ("temporal_transformer_lite", "temporal_transformer"):
            cfg_overrides["temporal_type"] = "transformer_lite"
        cfg = FishTemporalBaselineReIDConfig(**{**cfg_dict, **cfg_overrides})
        if num_classes is not None:
            cfg.num_classes = int(num_classes)
        return FishTemporalBaselineReID(cfg)

    if name in ("fishcnn", "cnn", "fishcnn_reid"):
        from fishmambatrack.models.reid.fishcnn_reid import FishCNNReID, FishCNNReIDConfig

        cfg = FishCNNReIDConfig(**cfg_dict)
        if num_classes is not None:
            cfg.num_classes = int(num_classes)
        return FishCNNReID(cfg)

    if name in ("fishtransformer", "transformer", "fishtransformer_reid"):
        from fishmambatrack.models.reid.fishtransformer_reid import FishTransformerReID, FishTransformerReIDConfig

        cfg = FishTransformerReIDConfig(**cfg_dict)
        if num_classes is not None:
            cfg.num_classes = int(num_classes)
        return FishTransformerReID(cfg)

    if name in ("resnet_bnneck", "resnet", "resnet_reid"):
        from fishmambatrack.models.reid.resnet_bnneck_reid import ResNetBNNeckReID, ResNetBNNeckReIDConfig

        cfg = ResNetBNNeckReIDConfig(**cfg_dict)
        if num_classes is not None:
            cfg.num_classes = int(num_classes)
        return ResNetBNNeckReID(cfg)

    raise ValueError(f"Unknown ReID model_name='{model_name}'.")


def load_reid_from_checkpoint(
    ckpt_path: str,
    *,
    device: torch.device,
    num_classes: Optional[int] = 0,
    model_name: Optional[str] = None,
) -> Tuple[torch.nn.Module, Dict[str, Any], Tuple[int, int]]:
    """
    Load ReID model and weights.

    Returns:
      model, meta_dict, (num_missing, num_unexpected)

    Notes:
      - If checkpoint has `meta.model_name`, it is used unless `model_name` is provided.
      - If checkpoint has no meta, defaults to fishmamba for backward compatibility.
    """
    ckpt_obj = torch.load(ckpt_path, map_location="cpu")
    meta_obj = _extract_meta(ckpt_obj)

    resolved_name = str(model_name or (meta_obj.model_name if meta_obj else "fishmamba"))
    resolved_cfg = meta_obj.model_cfg if meta_obj else {}

    model = build_reid_model(resolved_name, model_cfg=resolved_cfg, num_classes=num_classes).to(device)
    state = _extract_state_dict(ckpt_obj)
    missing, unexpected = model.load_state_dict(state, strict=False)

    meta_out = {
        "model_name": resolved_name,
        "model_cfg": dict(resolved_cfg),
    }
    if isinstance(ckpt_obj, dict):
        meta_raw = ckpt_obj.get("meta", None)
        if isinstance(meta_raw, dict):
            # Preserve any extra training metadata (e.g., input size, crop pad).
            for k, v in meta_raw.items():
                if k not in meta_out:
                    meta_out[k] = v
    return model, meta_out, (len(missing), len(unexpected))
