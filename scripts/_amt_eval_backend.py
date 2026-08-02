#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-configuration AMT-L48 evaluation utilities.

This backend exports AMT-L48 tracking results in MOTChallenge format and
runs the official TrackEval package. It intentionally contains only the final
AMT-L48 configuration and performs a fixed-protocol evaluation.
"""

from __future__ import annotations

import argparse
import configparser
import csv
import hashlib
import inspect
import math
import os
import pickle
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

RELEASE_ROOT = Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))

from fishmambatrack.tracking.sut_like_tracker import (  # noqa: E402
    Detection,
    FishIoUParams,
    SUTLikeTracker,
    SUTLikeTrackerConfig,
)

DATA_ROOT = "data/MFT25-train"
SEQ_NAMES = ["BT-001", "BT-003", "BT-005", "MSK-002", "PF-001", "SN-001", "SN-013", "SN-015"]
SPLIT = "val_half"
GT_FILE = "gt/gt_val_half.txt"
FULL_GT_FILE = "gt/gt.txt"
DET_FILE = "det/det_yolox_ckpt.txt"
IMG_DIR = "img1"
IMG_W, IMG_H = 1920, 1080
FRAME_OFFSET_FALLBACK_VAL_HALF = 1501

REID_CKPT = "checkpoints/amt_l48/reid_best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = True
EMB_BATCH = 128
NUM_WORKERS = 8
CACHE_VERSION = "amt_l48_release_v2"
CACHE_DIR = "results/amt_l48_trackeval/emb_cache"

REID_CROP_PAD = 0.10
FISHMAMBA_REID_INPUT_HW = (128, 256)
DEFAULT_REID_INPUT_HW = (128, 256)
VEL_R_S0 = 0.010
VEL_R_S1 = 0.050
VEL_IOU_MATCH = 0.20
VEL_IOU_MARGIN = 0.05
HISTORY_LINK_IOU_TH = 0.20
HISTORY_LINK_IOU_MARGIN = 0.05

TEMPORAL_EVAL_MODE = "history"
TEMPORAL_HISTORY_MIN_DEPTH = 1
TEMPORAL_HISTORY_SHORT_FALLBACK = "none"
TEMPORAL_HISTORY_PAD_MODE = "earliest"
TEMPORAL_HISTORY_POOL_MODE = "keep"
TEMPORAL_HISTORY_FUSE_REPEAT = True
TEMPORAL_HISTORY_FUSE_COS_TH = 0.95

FINAL_CFG_NAME = "AMT-L48"
FINAL_TRACKER_CONFIG = SUTLikeTrackerConfig(
    det_low_th=0.12,
    det_high_th=0.60,
    det_nms_iou=0.90,
    fishiou_th=0.25,
    w_fishiou=1.0,
    w_app=1.25,
    w_app_low=0.50,
    w_app_stage3=0.0,
    w_app_crowd=0.55,
    w_axis=0.05,
    reid_long_th=0.62,
    reid_long_fishiou_gate=0.05,
    use_confirmed_cascade=True,
    drop_unconfirmed_on_miss=True,
    stage2_use_app=True,
    stage3_update_emb=False,
    freeze_emb_in_crowd=True,
    emb_update_sim_th=0.45,
    emb_update_fishiou_th=0.40,
    emb_gain_high=1.0,
    emb_gain_low=1.0,
    max_age=30,
    min_hits=2,
    min_hits_score_gate=0.80,
    emb_bank_size=1,
    emb_momentum=1.0,
    inertia=0.90,
    sim_relu=True,
    fishiou_params=FishIoUParams(
        adaptive_central=True,
        ar_ref=2.0,
        ar_scale_min=0.6,
        ar_scale_max=1.4,
    ),
)

def _default_input_hw_for_model(model_name: str) -> Tuple[int, int]:
    name = str(model_name).lower()
    if name in ("fishmamba", "mamba", "fishmamba_reid"):
        return FISHMAMBA_REID_INPUT_HW
    return DEFAULT_REID_INPUT_HW


def _parse_hw(hw) -> Optional[Tuple[int, int]]:
    if hw is None:
        return None
    if isinstance(hw, (list, tuple)) and len(hw) == 2:
        return (int(hw[0]), int(hw[1]))
    if isinstance(hw, str):
        s = hw.strip().lower().replace("x", " ").replace(",", " ")
        parts = [p for p in s.split() if p]
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            return (int(parts[0]), int(parts[1]))
    return None


def make_reid_transform(input_hw: Tuple[int, int]) -> transforms.Compose:
    h, w = int(input_hw[0]), int(input_hw[1])
    return transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------
# MOT parsing utilities
# ---------------------------------------------------------------------

def read_mot_file(path: str) -> List[Tuple[int, int, float, float, float, float, float]]:
    """
    Read MOTChallenge format.
    Returns rows: (frame, id, x, y, w, h, score)
    """
    rows = []
    sep_re = re.compile(r"[\\s,]+")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            parts = [p for p in sep_re.split(line) if p]
            if len(parts) < 7:
                continue
            try:
                frame = int(float(parts[0]))
                tid = int(float(parts[1]))
                x = float(parts[2]); y = float(parts[3]); w = float(parts[4]); h = float(parts[5])
                score = float(parts[6])
            except Exception:
                continue
            rows.append((frame, tid, x, y, w, h, score))
    return rows


def read_gt_file(path: str) -> List[Tuple[int, int, float, float, float, float, float]]:
    """
    Read GT file and drop ignored regions (conf<=0).
    """
    rows = []
    for frame, tid, x, y, w, h, conf in read_mot_file(path):
        if conf <= 0:
            continue
        rows.append((frame, tid, x, y, w, h, conf))
    return rows


def _file_sig(path: str) -> str:
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return "missing"
    return f"m{int(st.st_mtime)}_s{int(st.st_size)}"


def _infer_frame_offset(split_gt_path: str, full_gt_path: str) -> int:
    """
    Infer offset such that global_frame = local_frame + offset.
    Uses full GT if available; otherwise uses the configured val-half offset.
    """
    if os.path.normpath(split_gt_path) == os.path.normpath(full_gt_path):
        return 0
    if not os.path.exists(full_gt_path):
        return FRAME_OFFSET_FALLBACK_VAL_HALF if SPLIT == "val_half" else 0
    try:
        from fishmambatrack.data.mot.mot_utils import infer_frame_offset_from_full_gt
        return int(infer_frame_offset_from_full_gt(split_gt_path, full_gt_path))
    except Exception:
        return FRAME_OFFSET_FALLBACK_VAL_HALF if SPLIT == "val_half" else 0


def _infer_det_is_global(det_frames: List[int], gt_frames: List[int]) -> bool:
    """
    Heuristic: if det spans a longer frame range than GT split, it's in global frame space.
    """
    if not det_frames or not gt_frames:
        return False
    return max(det_frames) > max(gt_frames)


def group_by_frame(rows: List[Tuple[int, int, float, float, float, float, float]]) -> Dict[int, List[Tuple[int, np.ndarray, float]]]:
    """
    frame -> list of (id, tlwh, score)
    """
    out: Dict[int, List[Tuple[int, np.ndarray, float]]] = {}
    for frame, tid, x, y, w, h, score in rows:
        out.setdefault(frame, []).append((tid, np.array([x, y, w, h], dtype=np.float32), float(score)))
    return out


def iou_matrix_tlwh(a_tlwh: np.ndarray, b_tlwh: np.ndarray) -> np.ndarray:
    if a_tlwh.size == 0 or b_tlwh.size == 0:
        return np.zeros((a_tlwh.shape[0], b_tlwh.shape[0]), dtype=np.float32)
    a = a_tlwh.copy()
    b = b_tlwh.copy()
    a[:, 2] = a[:, 0] + a[:, 2]
    a[:, 3] = a[:, 1] + a[:, 3]
    b[:, 2] = b[:, 0] + b[:, 2]
    b[:, 3] = b[:, 1] + b[:, 3]
    # compute iou
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)
    inter_w = np.clip(inter_x2 - inter_x1, 0.0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0.0, None)
    inter = inter_w * inter_h
    area_a = np.clip(ax2 - ax1, 0.0, None) * np.clip(ay2 - ay1, 0.0, None)
    area_b = np.clip(bx2 - bx1, 0.0, None) * np.clip(by2 - by1, 0.0, None)
    union = area_a + area_b - inter + 1e-9
    return (inter / union).astype(np.float32)


# ---------------------------------------------------------------------
# Detection threshold filtering
# ---------------------------------------------------------------------

@dataclass
class DetFilterConfig:
    score_min: float = 0.10


def filter_dets(
    dets: List[Tuple[int, np.ndarray, float]],
    cfg: DetFilterConfig,
) -> List[Tuple[int, np.ndarray, float]]:
    out = []
    for tid, tlwh, score in dets:
        if score < cfg.score_min:
            continue
        x, y, w, h = tlwh.tolist()
        if w <= 1 or h <= 1:
            continue

        out.append((tid, tlwh, score))
    return out


# ---------------------------------------------------------------------
# Axis / velocity guidance for embedding extraction (self-supervised from det motion)
# ---------------------------------------------------------------------

def _norm2(v: np.ndarray, eps: float = 1e-9) -> float:
    return float(np.sqrt(v[0] * v[0] + v[1] * v[1] + eps))


def _match_prev_to_cur_indices(
    prev_boxes: np.ndarray,
    cur_boxes: np.ndarray,
    *,
    iou_th: float = VEL_IOU_MATCH,
    iou_margin: float = VEL_IOU_MARGIN,
) -> np.ndarray:
    """
    One-step matching from current detections to previous-frame detections.

    Returns:
      prev_idx: (N_cur,) int32, where -1 means unmatched.
    """
    n_cur = int(cur_boxes.shape[0])
    prev_idx = np.full((n_cur,), -1, dtype=np.int32)
    if prev_boxes.size == 0 or cur_boxes.size == 0:
        return prev_idx

    iou = iou_matrix_tlwh(prev_boxes, cur_boxes)  # (P,N)
    cost = -(iou.astype(np.float32))
    cost[iou < float(iou_th)] = 1e6
    r, c = linear_sum_assignment(cost)
    for pi, ci in zip(r.tolist(), c.tolist()):
        if cost[pi, ci] >= 1e5:
            continue
        if float(iou_margin) > 0.0 and prev_boxes.shape[0] >= 2:
            ious = iou[:, ci]
            best = float(ious[pi])
            sec = float(np.partition(ious, -2)[-2])
            if (best - sec) < float(iou_margin):
                continue
        prev_idx[ci] = int(pi)
    return prev_idx


def estimate_det_velocity_axes(
    prev_boxes: np.ndarray,
    cur_boxes: np.ndarray,
    w_min: float,
    iou_th: float = VEL_IOU_MATCH,
    *,
    r_s0: float = VEL_R_S0,
    r_s1: float = VEL_R_S1,
    iou_margin: float = VEL_IOU_MARGIN,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate per-detection velocity-guided axis override (cos2θ,sin2θ) by matching to previous-frame detections.

    w_min is interpreted as a reliability threshold in [0,1], matching mot_reid's axis_weight.

    Returns:
      axis_override: (N,2) float32
      override_mask: (N,) bool
      reverse_mask: (N,) bool  (direction scan: reverse token order when velocity points to the opposite direction)
    """
    N = cur_boxes.shape[0]
    axis = np.zeros((N, 2), dtype=np.float32)
    override = np.zeros((N,), dtype=np.bool_)
    reverse = np.zeros((N,), dtype=np.bool_)

    if prev_boxes.size == 0 or cur_boxes.size == 0:
        return axis, override, reverse

    iou = iou_matrix_tlwh(prev_boxes, cur_boxes)  # (P,N)
    # Hungarian on -IoU to maximize IoU
    cost = -(iou.astype(np.float32))
    cost[iou < float(iou_th)] = 1e6
    r, c = linear_sum_assignment(cost)
    for pi, ci in zip(r.tolist(), c.tolist()):
        if cost[pi, ci] >= 1e5:
            continue
        if float(iou_margin) > 0.0 and prev_boxes.shape[0] >= 2:
            ious = iou[:, ci]
            best = float(ious[pi])
            sec = float(np.partition(ious, -2)[-2])
            if (best - sec) < float(iou_margin):
                continue

        # centers
        p = prev_boxes[pi]
        d = cur_boxes[ci]
        pc = p[:2] + p[2:] * 0.5
        dc = d[:2] + d[2:] * 0.5
        v = (dc - pc).astype(np.float32)
        # normalized speed by sqrt(area) (similar to vel_norm_mode="sqrt_area")
        speed = _norm2(v)
        area = float(max(d[2] * d[3], 1.0))
        speed_norm = speed / math.sqrt(area)

        # reliability in [0,1]
        if float(r_s1) > float(r_s0):
            w = (speed_norm - float(r_s0)) / (float(r_s1) - float(r_s0))
        else:
            w = 1.0 if speed_norm >= float(r_s0) else 0.0
        w = float(np.clip(w, 0.0, 1.0))

        if w < float(w_min):
            continue

        nv = v / (speed + 1e-6)
        # Axis representation is π-periodic: (cos2θ, sin2θ) so that θ and θ+π are equivalent.
        # This matches FishMambaReID axis_to_theta() which expects cos2θ/sin2θ.
        c2 = float(nv[0] * nv[0] - nv[1] * nv[1])
        s2 = float(2.0 * nv[0] * nv[1])
        axis[ci, 0] = c2
        axis[ci, 1] = s2
        override[ci] = True
        # Pick a consistent scan direction: reverse when velocity is in the "negative" half-plane.
        # (equivalent to flipping θ by +π under the π-periodic axis).
        reverse[ci] = bool((nv[0] < 0.0) or (abs(float(nv[0])) < 1e-6 and nv[1] < 0.0))
    return axis, override, reverse


# ---------------------------------------------------------------------
# ReID model loading (robust via introspection)
# ---------------------------------------------------------------------

def build_reid_model(num_classes: int, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    """
    Build ReID model from checkpoint (FishMamba + baselines).

    - If checkpoint contains `meta.model_name`, we will instantiate the matching model.
    - Otherwise (checkpoints without metadata), fall back to FishMambaReID for backward compatibility.
    """
    from fishmambatrack.models.reid.registry import load_reid_from_checkpoint

    model, meta, (n_missing, n_unexpected) = load_reid_from_checkpoint(
        ckpt_path,
        device=device,
        num_classes=None,  # use cfg.num_classes from ckpt meta if available; otherwise classifier is disabled
    )
    model._reid_meta = dict(meta)
    input_hw = _parse_hw(meta.get("input_hw")) if isinstance(meta, dict) else None
    if input_hw is None:
        input_hw = _default_input_hw_for_model(meta.get("model_name", ""))
    model._reid_input_hw = input_hw
    crop_pad = float(meta.get("crop_pad", REID_CROP_PAD)) if isinstance(meta, dict) else float(REID_CROP_PAD)
    model._reid_crop_pad = crop_pad
    seq_len = 1
    if isinstance(meta, dict):
        seq_len_raw = meta.get("seq_len", None)
        if seq_len_raw is None:
            cfg = meta.get("model_cfg", {})
            if isinstance(cfg, dict):
                seq_len_raw = cfg.get("infer_repeat_len", None)
        try:
            seq_len = max(1, int(seq_len_raw))
        except Exception:
            seq_len = 1
    model._reid_seq_len = int(seq_len)
    model._reid_vel_r_s0 = float(meta.get("vel_r_s0", VEL_R_S0)) if isinstance(meta, dict) else float(VEL_R_S0)
    model._reid_vel_r_s1 = float(meta.get("vel_r_s1", VEL_R_S1)) if isinstance(meta, dict) else float(VEL_R_S1)
    print(f"[amt_l48] Loaded ckpt: {ckpt_path}")
    print(f"  ReID model: {meta.get('model_name', 'unknown')}")
    print(f"  Missing keys: {n_missing}")
    print(f"  Unexpected keys: {n_unexpected}")
    print(f"  ReID input_hw: {model._reid_input_hw}  crop_pad: {model._reid_crop_pad:.2f}  seq_len: {model._reid_seq_len}")

    try:
        fwd_sig = inspect.signature(model.forward)
        supported = [k for k in ["axis_override", "override_mask", "reverse_mask"] if k in fwd_sig.parameters]
        print(f"  Forward supports: {supported}")
    except Exception:
        pass

    model.eval()
    return model


def configure_temporal_infer_pool_mode(
    model: torch.nn.Module,
    *,
    temporal_eval_mode: str,
    history_pool_mode: str,
) -> None:
    """
    Optionally override temporal pooling mode at inference time.
    """
    cfg = getattr(model, "cfg", None)
    if cfg is None or (not hasattr(cfg, "pool_mode")):
        return

    cur_mode = str(getattr(cfg, "pool_mode", "mean_last")).lower()
    temporal_eval_mode = str(temporal_eval_mode).lower()
    mode = str(history_pool_mode).lower()
    if mode not in {"auto", "keep", "last", "mean", "mean_last"}:
        raise ValueError(f"Unknown history_pool_mode: {history_pool_mode}")

    target_mode = cur_mode
    if mode == "auto":
        if temporal_eval_mode == "history":
            target_mode = "last"
    elif mode == "keep":
        target_mode = cur_mode
    else:
        target_mode = mode

    if target_mode != cur_mode:
        setattr(cfg, "pool_mode", target_mode)
        print(f"[amt_l48] temporal pool_mode override: {cur_mode} -> {target_mode}")
    else:
        print(f"[amt_l48] temporal pool_mode: {cur_mode}")


def forward_to_embedding(model: torch.nn.Module, x: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Supports both dict-output and tuple-output models.
    """
    out = model(x, **kwargs)
    if isinstance(out, dict):
        # common: {"emb": ..., "axis": ...}
        if "emb" in out:
            return out["emb"]
        # fallback: first tensor in dict
        for v in out.values():
            if torch.is_tensor(v) and v.dim() == 2:
                return v
        raise RuntimeError("Model output dict does not contain emb.")
    if isinstance(out, (list, tuple)):
        # common: (emb, axis, ...)
        for v in out:
            if torch.is_tensor(v) and v.dim() == 2:
                return v
        raise RuntimeError("Model output tuple/list has no 2D embedding tensor.")
    raise RuntimeError(f"Unsupported model output type: {type(out)}")


def forward_to_emb_and_axis(model: torch.nn.Module, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Returns (emb, axis_used) if available.
    - For FishMambaReID dict output: uses out["axis_used"] (fallback out["axis"])
    - For tuple/list output: tries to find (B,2) tensor as axis
    """
    out = model(x, **kwargs)
    if isinstance(out, dict):
        if "emb" in out:
            emb = out["emb"]
        else:
            emb = None
            for v in out.values():
                if torch.is_tensor(v) and v.dim() == 2 and v.shape[1] > 2:
                    emb = v
                    break
            if emb is None:
                raise RuntimeError("Model output dict does not contain emb.")

        axis = out.get("axis_used", None)
        if axis is None:
            axis = out.get("axis", None)
        if axis is not None and not torch.is_tensor(axis):
            axis = None
        return emb, axis

    if isinstance(out, (list, tuple)):
        emb = None
        axis = None
        for v in out:
            if not torch.is_tensor(v) or v.dim() != 2:
                continue
            if v.shape[1] == 2 and axis is None:
                axis = v
            elif v.shape[1] > 2 and emb is None:
                emb = v
        if emb is None:
            raise RuntimeError("Model output tuple/list has no 2D embedding tensor.")
        return emb, axis

    raise RuntimeError(f"Unsupported model output type: {type(out)}")


# ---------------------------------------------------------------------
# Embedding cache build
# ---------------------------------------------------------------------


def crop_tlwh(img: Image.Image, tlwh: np.ndarray, *, pad_ratio: float = REID_CROP_PAD) -> Image.Image:
    """Crop tlwh from PIL image with padding and boundary clamping."""
    img_w, img_h = img.size
    x, y, w, h = tlwh.tolist()
    padx = float(w) * float(pad_ratio)
    pady = float(h) * float(pad_ratio)
    x1 = int(max(0, math.floor(x - padx)))
    y1 = int(max(0, math.floor(y - pady)))
    x2 = int(min(img_w - 1, math.ceil(x + w + padx)))
    y2 = int(min(img_h - 1, math.ceil(y + h + pady)))
    if x2 <= x1 or y2 <= y1:
        x2 = min(img_w - 1, x1 + 2)
        y2 = min(img_h - 1, y1 + 2)
    return img.crop((x1, y1, x2, y2))


def _collate_first(batch):
    return batch[0]


class _FrameChunkDataset(Dataset):
    def __init__(self, seq_dir: str, items: List[dict], transform, *, crop_pad: float):
        self.seq_dir = seq_dir
        self.items = items
        self.transform = transform
        self.crop_pad = float(crop_pad)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        it = self.items[idx]
        local_frame = int(it["local_frame"])
        global_frame = int(it["global_frame"])
        i0 = int(it["i0"])

        boxes = it["boxes"]  # (K,4) float32
        scores = it["scores"]  # (K,) float32
        axis_override = it.get("axis_override", None)  # (K,2) float32
        override_mask = it.get("override_mask", None)  # (K,) bool
        reverse_mask = it.get("reverse_mask", None)  # (K,) bool

        img_path = os.path.join(self.seq_dir, IMG_DIR, f"{global_frame:06d}.jpg")
        hist_global_frames = it.get("hist_global_frames", None)  # (K,T) int32
        hist_boxes = it.get("hist_boxes", None)  # (K,T,4) float32

        if hist_global_frames is not None and hist_boxes is not None:
            h_frames = np.asarray(hist_global_frames, dtype=np.int32)
            h_boxes = np.asarray(hist_boxes, dtype=np.float32)
            # Load only the frames needed by this chunk once.
            frame_images: Dict[int, Image.Image] = {}
            for gf in np.unique(h_frames):
                p = os.path.join(self.seq_dir, IMG_DIR, f"{int(gf):06d}.jpg")
                with Image.open(p) as _im:
                    frame_images[int(gf)] = _im.convert("RGB")

            seq_crops: List[torch.Tensor] = []
            k_det, t_len = h_frames.shape
            for k in range(k_det):
                ts: List[torch.Tensor] = []
                for t in range(t_len):
                    gf = int(h_frames[k, t])
                    ts.append(self.transform(crop_tlwh(frame_images[gf], h_boxes[k, t], pad_ratio=self.crop_pad)))
                seq_crops.append(torch.stack(ts, dim=0))  # (T,3,H,W)
            x = torch.stack(seq_crops, dim=0)  # (K,T,3,H,W) on CPU
        else:
            with Image.open(img_path) as _im:
                img = _im.convert("RGB")
            crops = [self.transform(crop_tlwh(img, b, pad_ratio=self.crop_pad)) for b in boxes]
            x = torch.stack(crops, dim=0)  # (K,3,H,W) on CPU

        return {
            "local_frame": local_frame,
            "global_frame": global_frame,
            "i0": i0,
            "boxes": torch.from_numpy(boxes),
            "scores": torch.from_numpy(scores),
            "axis_override": None if axis_override is None else torch.from_numpy(axis_override),
            "override_mask": None if override_mask is None else torch.from_numpy(override_mask.astype(np.bool_)),
            "reverse_mask": None if reverse_mask is None else torch.from_numpy(reverse_mask.astype(np.bool_)),
            "x": x,
        }


def _build_temporal_chunk_histories(
    *,
    local_frame: int,
    frame_offset: int,
    cur_boxes: np.ndarray,
    i0: int,
    i1: int,
    seq_len: int,
    boxes_by_local: Dict[int, np.ndarray],
    prev_idx_by_local: Dict[int, np.ndarray],
    min_depth_to_keep: int = 1,
    short_fallback: str = "none",
    pad_mode: str = "earliest",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build temporal crop plans for one chunk via backward one-step links.
    """
    k = int(i1 - i0)
    t_len = int(max(1, seq_len))
    min_depth_to_keep = int(max(1, min_depth_to_keep))
    short_fallback = str(short_fallback).lower()
    if short_fallback not in {"none", "repeat"}:
        raise ValueError(f"Unknown short_fallback: {short_fallback}")
    pad_mode = str(pad_mode).lower()
    if pad_mode not in {"earliest", "current"}:
        raise ValueError(f"Unknown pad_mode: {pad_mode}")

    hist_global_frames = np.zeros((k, t_len), dtype=np.int32)
    hist_boxes = np.zeros((k, t_len, 4), dtype=np.float32)

    for j in range(k):
        det_idx = int(i0 + j)
        cur_box = cur_boxes[det_idx]
        cur_local = int(local_frame)
        cur_det = int(det_idx)

        g_now = int(cur_local + frame_offset)
        hist_global_frames[j, :] = g_now
        hist_boxes[j, :, :] = cur_box

        write_pos = t_len - 1
        valid_depth = 1
        while write_pos > 0:
            prev_idx = prev_idx_by_local.get(cur_local, None)
            if prev_idx is None or cur_det < 0 or cur_det >= int(prev_idx.shape[0]):
                break
            p_det = int(prev_idx[cur_det])
            if p_det < 0:
                break
            p_local = int(cur_local - 1)
            p_boxes = boxes_by_local.get(p_local, None)
            if p_boxes is None or p_det >= int(p_boxes.shape[0]):
                break

            write_pos -= 1
            hist_global_frames[j, write_pos] = int(p_local + frame_offset)
            hist_boxes[j, write_pos] = p_boxes[p_det]
            cur_local = p_local
            cur_det = p_det
            valid_depth += 1

        if short_fallback == "repeat" and valid_depth < min_depth_to_keep:
            hist_global_frames[j, :] = g_now
            hist_boxes[j, :, :] = cur_box
            continue

        # Left-pad with the selected policy.
        if write_pos > 0:
            if pad_mode == "current":
                hist_global_frames[j, :write_pos] = g_now
                hist_boxes[j, :write_pos] = cur_box
            else:
                hist_global_frames[j, :write_pos] = hist_global_frames[j, write_pos]
                hist_boxes[j, :write_pos] = hist_boxes[j, write_pos]

    return hist_global_frames, hist_boxes


def build_embedding_cache_for_seq(
    seq_dir: str,
    seq_name: str,
    det_by_frame: Dict[int, List[Tuple[int, np.ndarray, float]]],
    det_path: str,
    gt_min_frame: int,
    gt_max_frame: int,
    frame_offset: int,
    det_is_global: bool,
    model: torch.nn.Module,
    axis_mode: str,
    w_min: Optional[float],
    tracker_cfg: SUTLikeTrackerConfig,
    det_filter: DetFilterConfig,
    *,
    reid_ckpt_path: Optional[str] = None,
    reid_input_hw: Optional[Tuple[int, int]] = None,
    reid_crop_pad: Optional[float] = None,
    temporal_eval_mode: str = "history",
    history_min_depth: int = 1,
    history_short_fallback: str = "none",
    history_pad_mode: str = "earliest",
    history_fuse_repeat: bool = False,
    history_fuse_cos_th: float = 0.55,
    history_link_iou_th: float = HISTORY_LINK_IOU_TH,
    history_link_iou_margin: float = HISTORY_LINK_IOU_MARGIN,
) -> Dict[int, List[Detection]]:
    """
    Returns: local_frame -> List[Detection] with embeddings
    """
    ckpt_path_for_cache = REID_CKPT if reid_ckpt_path is None else str(reid_ckpt_path)
    input_hw = _parse_hw(reid_input_hw) or getattr(model, "_reid_input_hw", None)
    if input_hw is None:
        input_hw = DEFAULT_REID_INPUT_HW
    crop_pad = float(reid_crop_pad) if reid_crop_pad is not None else float(getattr(model, "_reid_crop_pad", REID_CROP_PAD))
    vel_r_s0 = float(getattr(model, "_reid_vel_r_s0", VEL_R_S0))
    vel_r_s1 = float(getattr(model, "_reid_vel_r_s1", VEL_R_S1))
    temporal_seq_len = int(max(1, getattr(model, "_reid_seq_len", 1)))
    temporal_eval_mode = str(temporal_eval_mode).lower()
    if temporal_eval_mode not in {"history", "repeat"}:
        raise ValueError(f"Unknown temporal_eval_mode: {temporal_eval_mode}")
    history_min_depth = int(max(1, history_min_depth))
    history_short_fallback = str(history_short_fallback).lower()
    if history_short_fallback not in {"none", "repeat"}:
        raise ValueError(f"Unknown history_short_fallback: {history_short_fallback}")
    history_pad_mode = str(history_pad_mode).lower()
    if history_pad_mode not in {"earliest", "current"}:
        raise ValueError(f"Unknown history_pad_mode: {history_pad_mode}")
    history_fuse_repeat = bool(history_fuse_repeat)
    history_fuse_cos_th = float(history_fuse_cos_th)
    history_link_iou_th = float(history_link_iou_th)
    history_link_iou_margin = float(history_link_iou_margin)
    if not (0.0 <= history_link_iou_th <= 1.0):
        raise ValueError(f"history_link_iou_th out of range: {history_link_iou_th}")
    if history_link_iou_margin < 0.0:
        raise ValueError(f"history_link_iou_margin must be >=0: {history_link_iou_margin}")
    pool_mode = str(getattr(getattr(model, "cfg", object()), "pool_mode", "na"))
    use_temporal_history = (temporal_seq_len > 1) and (temporal_eval_mode == "history")
    emb_batch_eff = int(max(1, EMB_BATCH // temporal_seq_len)) if use_temporal_history else int(EMB_BATCH)
    ckpt_tag = os.path.basename(ckpt_path_for_cache).replace(".pt", "").replace(".pth", "")
    w_tag = "None" if w_min is None else f"{w_min:.2f}"

    cache_key = "|".join([
        CACHE_VERSION,
        seq_name,
        os.path.basename(GT_FILE),
        f"gt={gt_min_frame}-{gt_max_frame}",
        f"off={frame_offset}",
        f"det_global={int(det_is_global)}",
        f"det={os.path.normpath(det_path)}",
        f"det_sig={_file_sig(det_path)}",
        f"ckpt={os.path.normpath(ckpt_path_for_cache)}",
        f"ckpt_sig={_file_sig(ckpt_path_for_cache)}",
        f"axis_mode={axis_mode}",
        f"w_min={w_tag}",
        f"reid_hw={input_hw[0]}x{input_hw[1]}",
        f"crop_pad={crop_pad:.3f}",
        f"seq_len={temporal_seq_len}",
        f"temporal_mode={temporal_eval_mode}",
        f"pool_mode={pool_mode}",
        f"hist_min_depth={history_min_depth}",
        f"hist_short_fb={history_short_fallback}",
        f"hist_pad={history_pad_mode}",
        f"hist_fuse_repeat={int(history_fuse_repeat)}",
        f"hist_fuse_cos_th={history_fuse_cos_th:.2f}",
        f"hist_link_iou={history_link_iou_th:.2f}",
        f"hist_link_margin={history_link_iou_margin:.2f}",
        f"vel_r={vel_r_s0:.3f}-{vel_r_s1:.3f}",
        f"vel_iou={VEL_IOU_MATCH:.2f}-{VEL_IOU_MARGIN:.2f}",
        f"det_score_min={det_filter.score_min:.3f}",
    ])
    cache_id = hashlib.md5(cache_key.encode("utf-8")).hexdigest()[:12]
    cache_path = os.path.join(CACHE_DIR, f"{seq_name}__{cache_id}.pkl")

    if os.path.exists(cache_path):
        print(f"  [cache] {seq_name}: loading -> {cache_path}", flush=True)
        with open(cache_path, "rb") as f:
            obj = pickle.load(f)
        print(f"  [cache] {seq_name}: loaded", flush=True)
        return obj

    num_frames = int(gt_max_frame - gt_min_frame + 1)
    print(f"  [cache] {seq_name}: building {cache_id} frames={num_frames} local={gt_min_frame}..{gt_max_frame} offset={frame_offset} det_is_global={det_is_global}")
    print(
        f"  [cache] {seq_name}: reid_hw={input_hw} crop_pad={crop_pad:.2f} "
        f"seq_len={temporal_seq_len} temporal_mode={temporal_eval_mode} pool_mode={pool_mode} "
        f"hist_min_depth={history_min_depth} hist_short_fb={history_short_fallback} hist_pad={history_pad_mode} "
        f"hist_fuse_repeat={int(history_fuse_repeat)} hist_fuse_cos_th={history_fuse_cos_th:.2f} "
        f"hist_link_iou={history_link_iou_th:.2f} hist_link_margin={history_link_iou_margin:.2f} "
        f"emb_batch_eff={emb_batch_eff} "
        f"vel_r=({vel_r_s0:.3f},{vel_r_s1:.3f})"
    )

    # Build cache skeleton + chunk jobs (CPU)
    cache: Dict[int, List[Detection]] = {}
    chunk_items: List[dict] = []
    boxes_by_local: Dict[int, np.ndarray] = {}
    prev_idx_by_local: Dict[int, np.ndarray] = {}
    prev_boxes = np.zeros((0, 4), dtype=np.float32)
    total_dets = 0

    for local_frame in range(gt_min_frame, gt_max_frame + 1):
        global_frame = local_frame + frame_offset
        det_frame = global_frame if det_is_global else local_frame
        dets_raw = det_by_frame.get(det_frame, [])
        dets_raw = filter_dets(dets_raw, det_filter)

        if len(dets_raw) == 0:
            cache[local_frame] = []
            prev_boxes = np.zeros((0, 4), dtype=np.float32)
            continue

        cur_boxes = np.stack([tlwh for _, tlwh, _ in dets_raw], axis=0).astype(np.float32)
        scores = np.array([score for _, _, score in dets_raw], dtype=np.float32)
        total_dets += int(cur_boxes.shape[0])
        cur_prev_idx = _match_prev_to_cur_indices(
            prev_boxes,
            cur_boxes,
            iou_th=history_link_iou_th,
            iou_margin=history_link_iou_margin,
        )
        boxes_by_local[local_frame] = cur_boxes
        prev_idx_by_local[local_frame] = cur_prev_idx

        # Axis override computation (velocity)
        axis_override = None
        override_mask = None
        reverse_mask = None

        if axis_mode == "none":
            axis_override = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (cur_boxes.shape[0], 1))
            override_mask = np.ones((cur_boxes.shape[0],), dtype=np.bool_)
            reverse_mask = np.zeros((cur_boxes.shape[0],), dtype=np.bool_)
        elif axis_mode in ("vel_or_pred", "vel_scan_or_pred"):
            assert w_min is not None
            ax, mk, rv = estimate_det_velocity_axes(
                prev_boxes,
                cur_boxes,
                w_min=w_min,
                iou_th=VEL_IOU_MATCH,
                r_s0=vel_r_s0,
                r_s1=vel_r_s1,
                iou_margin=VEL_IOU_MARGIN,
            )
            axis_override = ax
            override_mask = mk
            reverse_mask = rv if axis_mode == "vel_scan_or_pred" else np.zeros_like(rv)
        elif axis_mode == "pred":
            axis_override = None
            override_mask = None
            reverse_mask = None
        else:
            raise ValueError(f"Unknown axis_mode: {axis_mode}")

        N = int(cur_boxes.shape[0])
        cache[local_frame] = [None] * N  # type: ignore[list-item]
        for i0 in range(0, N, emb_batch_eff):
            i1 = min(N, i0 + emb_batch_eff)
            hist_global_frames = None
            hist_boxes = None
            if use_temporal_history:
                hist_global_frames, hist_boxes = _build_temporal_chunk_histories(
                    local_frame=local_frame,
                    frame_offset=frame_offset,
                    cur_boxes=cur_boxes,
                    i0=i0,
                    i1=i1,
                    seq_len=temporal_seq_len,
                    boxes_by_local=boxes_by_local,
                    prev_idx_by_local=prev_idx_by_local,
                    min_depth_to_keep=history_min_depth,
                    short_fallback=history_short_fallback,
                    pad_mode=history_pad_mode,
                )
            chunk_items.append({
                "local_frame": local_frame,
                "global_frame": global_frame,
                "i0": i0,
                "boxes": cur_boxes[i0:i1],
                "scores": scores[i0:i1],
                "axis_override": None if axis_override is None else axis_override[i0:i1],
                "override_mask": None if override_mask is None else override_mask[i0:i1],
                "reverse_mask": None if reverse_mask is None else reverse_mask[i0:i1],
                "hist_global_frames": hist_global_frames,
                "hist_boxes": hist_boxes,
            })

        prev_boxes = cur_boxes

    # No detections at all (edge case)
    if len(chunk_items) == 0:
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        print(f"  [cache] {seq_name}: saved -> {cache_path}")
        return cache

    # Multi-process CPU prefetch for image decode + crop/transform
    device = next(model.parameters()).device
    fwd_sig = inspect.signature(model.forward)
    pin_memory = (device.type == "cuda") and (not use_temporal_history)
    loader_workers = int(NUM_WORKERS)
    if use_temporal_history:
        # Large (K,T,C,H,W) samples with multiprocessing can hit file-handle limits.
        loader_workers = 0

    transform = make_reid_transform(input_hw)
    ds = _FrameChunkDataset(seq_dir=seq_dir, items=chunk_items, transform=transform, crop_pad=crop_pad)
    dl_kwargs = dict(
        batch_size=1,
        shuffle=False,
        num_workers=loader_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_first,
    )
    if loader_workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = 1 if use_temporal_history else 2

    loader = DataLoader(ds, **dl_kwargs)

    t0 = time.time()
    last_log = t0
    chunks_done = 0
    dets_done = 0
    total_chunks = len(chunk_items)

    for it in loader:
        chunks_done += 1

        x = it["x"].to(device, non_blocking=True)
        axis_override_t = it.get("axis_override", None)
        override_mask_t = it.get("override_mask", None)
        reverse_mask_t = it.get("reverse_mask", None)

        fwd_kwargs = {}
        if "return_logits" in fwd_sig.parameters:
            fwd_kwargs["return_logits"] = False
        if axis_override_t is not None and "axis_override" in fwd_sig.parameters:
            fwd_kwargs["axis_override"] = axis_override_t.to(device, non_blocking=True)
        if override_mask_t is not None and "override_mask" in fwd_sig.parameters:
            fwd_kwargs["override_mask"] = override_mask_t.to(device, non_blocking=True)
        if reverse_mask_t is not None and "reverse_mask" in fwd_sig.parameters:
            fwd_kwargs["reverse_mask"] = reverse_mask_t.to(device, non_blocking=True)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=USE_AMP and device.type == "cuda"):
                emb_t, axis_t = forward_to_emb_and_axis(model, x, **fwd_kwargs)
                if use_temporal_history and history_fuse_repeat and x.ndim == 5:
                    # Robustify history inference: if history embedding diverges too much from
                    # single-frame embedding, fallback to single-frame for that detection.
                    x_last = x[:, -1, :, :, :]
                    emb_rep = forward_to_embedding(model, x_last)
                    h = F.normalize(emb_t.float(), dim=1)
                    r = F.normalize(emb_rep.float(), dim=1)
                    cos = (h * r).sum(dim=1, keepdim=True)
                    use_rep = (cos < float(history_fuse_cos_th)).to(h.dtype)
                    emb_t = F.normalize((1.0 - use_rep) * h + use_rep * r, dim=1)

        emb = emb_t.float().cpu().numpy().astype(np.float32, copy=False)
        axis = axis_t.float().cpu().numpy().astype(np.float32, copy=False) if axis_t is not None else None

        local_frame = int(it["local_frame"])
        i0 = int(it["i0"])
        boxes = it["boxes"].numpy().astype(np.float32, copy=False)
        scores = it["scores"].numpy().astype(np.float32, copy=False)

        K = int(boxes.shape[0])
        dets_done += K
        for k in range(K):
            ax_i = None if axis is None else axis[k]
            cache[local_frame][i0 + k] = Detection(
                tlwh=boxes[k],
                score=float(scores[k]),
                emb=emb[k],
                axis=None if ax_i is None else ax_i,
            )

        now = time.time()
        if chunks_done == 1 or chunks_done == total_chunks or (chunks_done % 200 == 0) or (now - last_log) >= 15.0:
            dt = max(1e-6, now - t0)
            rate = dets_done / dt
            eta_s = (total_dets - dets_done) / max(1e-6, rate)
            print(f"  [cache:{seq_name}] chunks {chunks_done}/{total_chunks}  dets {dets_done}/{total_dets}  {rate:.1f} det/s  ETA={eta_s/60:.1f} min")
            last_log = now

    # Verify all slots filled
    for fr, lst in cache.items():
        if lst and any(d is None for d in lst):
            raise RuntimeError(f"[{seq_name}] cache incomplete at frame={fr}")

    # Save cache
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    print(f"  [cache] {seq_name}: saved -> {cache_path}")

    return cache


# ---------------------------------------------------------------------
# Evaluation (CLEAR MOT + IDF1)
# ---------------------------------------------------------------------



def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else RELEASE_ROOT / p


def _write_seqinfo(src_seq_dir: Path, dst_seq_dir: Path, seq_name: str, seq_length: int) -> None:
    cp = configparser.ConfigParser()
    cp.optionxform = str
    src = src_seq_dir / "seqinfo.ini"
    if src.exists():
        cp.read(src)
    if "Sequence" not in cp:
        cp["Sequence"] = {}
    sec = cp["Sequence"]
    sec["name"] = seq_name
    sec["imDir"] = sec.get("imDir", "img1")
    sec["frameRate"] = sec.get("frameRate", "25")
    sec["seqLength"] = str(int(seq_length))
    sec["imWidth"] = sec.get("imWidth", str(IMG_W))
    sec["imHeight"] = sec.get("imHeight", str(IMG_H))
    sec["imExt"] = sec.get("imExt", ".jpg")
    dst_seq_dir.mkdir(parents=True, exist_ok=True)
    with (dst_seq_dir / "seqinfo.ini").open("w", encoding="utf-8") as f:
        cp.write(f, space_around_delimiters=False)


def prepare_trackeval_gt(data_root: Path, gt_root: Path, seqmap_file: Path) -> Dict[str, Tuple[int, int]]:
    gt_root.mkdir(parents=True, exist_ok=True)
    seqmap_file.parent.mkdir(parents=True, exist_ok=True)
    seqmap_file.write_text("name\n" + "\n".join(SEQ_NAMES) + "\n", encoding="utf-8")
    seq_ranges: Dict[str, Tuple[int, int]] = {}
    for seq in SEQ_NAMES:
        src_seq_dir = data_root / seq
        src_gt = src_seq_dir / GT_FILE
        if not src_gt.exists():
            raise FileNotFoundError(src_gt)
        rows = read_gt_file(str(src_gt))
        if not rows:
            raise RuntimeError(f"Empty GT: {src_gt}")
        frames = [int(r[0]) for r in rows]
        gt_min, gt_max = min(frames), max(frames)
        seq_ranges[seq] = (gt_min, gt_max)
        dst_seq_dir = gt_root / seq
        (dst_seq_dir / "gt").mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_gt, dst_seq_dir / "gt" / "gt.txt")
        _write_seqinfo(src_seq_dir, dst_seq_dir, seq, gt_max)
    return seq_ranges


def write_mot_predictions(out_file: Path, pred_by_frame: Dict[int, List[Tuple[int, np.ndarray]]]) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for frame_id in sorted(pred_by_frame):
            for track_id, tlwh in pred_by_frame[frame_id]:
                x, y, w, h = [float(v) for v in tlwh]
                f.write(f"{int(frame_id)},{int(track_id)},{x:.3f},{y:.3f},{w:.3f},{h:.3f},1\n")


def export_amt_l48_mot(data_root: Path, checkpoint: Path, output_dir: Path, *, device: str, amp: bool, emb_batch: int, num_workers: int) -> None:
    global DATA_ROOT, REID_CKPT, DEVICE, USE_AMP, EMB_BATCH, NUM_WORKERS, CACHE_DIR
    DATA_ROOT = str(data_root)
    REID_CKPT = str(checkpoint)
    DEVICE = str(device)
    USE_AMP = bool(amp) and DEVICE.startswith("cuda")
    EMB_BATCH = int(emb_batch)
    NUM_WORKERS = int(num_workers)
    CACHE_DIR = str(output_dir / "emb_cache")
    os.makedirs(CACHE_DIR, exist_ok=True)

    gt_root = output_dir / "trackeval_data" / "gt" / "mot_challenge" / "MFT25-val"
    tracker_root = output_dir / "trackeval_data" / "trackers" / "mot_challenge" / "MFT25-val" / "AMT_L48" / "data"
    seqmap_file = output_dir / "trackeval_data" / "gt" / "mot_challenge" / "seqmaps" / "MFT25-val.txt"
    mot_root = output_dir / "mot_results" / "AMT_L48"
    seq_ranges = prepare_trackeval_gt(data_root, gt_root, seqmap_file)

    torch_device = torch.device(DEVICE)
    model = build_reid_model(num_classes=0, ckpt_path=str(checkpoint), device=torch_device)
    configure_temporal_infer_pool_mode(
        model,
        temporal_eval_mode=TEMPORAL_EVAL_MODE,
        history_pool_mode=TEMPORAL_HISTORY_POOL_MODE,
    )
    cfg = FINAL_TRACKER_CONFIG
    det_filter = DetFilterConfig(score_min=float(cfg.det_low_th))
    fishiou_params = cfg.fishiou_params or FishIoUParams(alpha=0.15, beta=0.30, gamma=0.25, w1=1.0, w2=0.3, w3=0.1, w4=0.2, w5=0.4)

    manifest_rows = []
    for seq in SEQ_NAMES:
        seq_dir = data_root / seq
        gt_path = seq_dir / GT_FILE
        full_gt_path = seq_dir / FULL_GT_FILE
        det_path = seq_dir / DET_FILE
        gt_rows = read_gt_file(str(gt_path))
        gt_by_frame = group_by_frame(gt_rows)
        gt_min_frame, gt_max_frame = seq_ranges[seq]
        frame_offset = 0 if gt_min_frame != 1 else int(_infer_frame_offset(str(gt_path), str(full_gt_path)))
        det_rows = read_mot_file(str(det_path))
        det_by_frame = group_by_frame(det_rows)
        det_is_global = _infer_det_is_global(list(det_by_frame.keys()), list(gt_by_frame.keys()))

        cache = build_embedding_cache_for_seq(
            seq_dir=str(seq_dir),
            seq_name=seq,
            det_by_frame=det_by_frame,
            det_path=str(det_path),
            gt_min_frame=int(gt_min_frame),
            gt_max_frame=int(gt_max_frame),
            frame_offset=int(frame_offset),
            det_is_global=bool(det_is_global),
            model=model,
            axis_mode="none",
            w_min=None,
            tracker_cfg=cfg,
            det_filter=det_filter,
            reid_ckpt_path=str(checkpoint),
            temporal_eval_mode=TEMPORAL_EVAL_MODE,
            history_min_depth=TEMPORAL_HISTORY_MIN_DEPTH,
            history_short_fallback=TEMPORAL_HISTORY_SHORT_FALLBACK,
            history_pad_mode=TEMPORAL_HISTORY_PAD_MODE,
            history_fuse_repeat=TEMPORAL_HISTORY_FUSE_REPEAT,
            history_fuse_cos_th=TEMPORAL_HISTORY_FUSE_COS_TH,
            history_link_iou_th=HISTORY_LINK_IOU_TH,
            history_link_iou_margin=HISTORY_LINK_IOU_MARGIN,
        )

        tracker = SUTLikeTracker(cfg, fishiou_params=fishiou_params)
        pred_by_frame: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        for frame_id in range(int(gt_min_frame), int(gt_max_frame) + 1):
            tracks = tracker.update(cache.get(frame_id, []))
            pred_by_frame[frame_id] = [(int(t.track_id), t.tlwh.copy()) for t in tracks]

        write_mot_predictions(mot_root / f"{seq}.txt", pred_by_frame)
        write_mot_predictions(tracker_root / f"{seq}.txt", pred_by_frame)
        n_pred = sum(len(v) for v in pred_by_frame.values())
        manifest_rows.append((seq, gt_min_frame, gt_max_frame, frame_offset, int(det_is_global), n_pred))
        print(f"[export] {seq} frames={gt_min_frame}..{gt_max_frame} predictions={n_pred}")

    with (output_dir / "export_manifest.csv").open("w", encoding="utf-8") as f:
        f.write("sequence,gt_min_frame,gt_max_frame,frame_offset,det_is_global,num_predictions\n")
        for row in manifest_rows:
            f.write(",".join(map(str, row)) + "\n")


def _import_official_trackeval(trackeval_root: Optional[Path]):
    if trackeval_root is not None:
        sys.path.insert(0, str(trackeval_root))
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]
    try:
        import trackeval  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Official TrackEval is required. Install it from "
            "https://github.com/JonathonLuiten/TrackEval or pass --trackeval_root."
        ) from exc
    return trackeval


def _write_summary_csv(output_dir: Path) -> None:
    summary = output_dir / "trackeval_raw_outputs" / "AMT_L48" / "pedestrian_summary.txt"
    detailed = output_dir / "trackeval_raw_outputs" / "AMT_L48" / "pedestrian_detailed.csv"
    lines = [ln.strip() for ln in summary.read_text(encoding="utf-8").splitlines() if ln.strip()]
    fields = lines[0].split()
    values = lines[1].split()
    row = dict(zip(fields, values))
    main = {
        "method": "AMT-L48",
        "evaluator": "official TrackEval",
        "HOTA": row.get("HOTA", ""),
        "DetA": row.get("DetA", ""),
        "AssA": row.get("AssA", ""),
        "IDF1": row.get("IDF1", ""),
        "MOTA": row.get("MOTA", ""),
        "IDSW": row.get("IDSW", ""),
        "FP": row.get("CLR_FP", ""),
        "FN": row.get("CLR_FN", ""),
        "Frag": row.get("Frag", ""),
    }
    with (output_dir / "official_trackeval_main_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(main.keys()))
        writer.writeheader()
        writer.writerow(main)

    def pct(r: dict, key: str) -> str:
        val = r.get(key, "")
        return "" if val == "" else f"{100.0 * float(val):.3f}"

    with detailed.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out_fields = ["sequence", "HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW", "FP", "FN", "Frag"]
    with (output_dir / "official_trackeval_per_sequence_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for r in rows:
            seq = r.get("seq", "")
            if not seq:
                continue
            writer.writerow({
                "sequence": seq,
                "HOTA": pct(r, "HOTA___AUC"),
                "DetA": pct(r, "DetA___AUC"),
                "AssA": pct(r, "AssA___AUC"),
                "IDF1": pct(r, "IDF1"),
                "MOTA": pct(r, "MOTA"),
                "IDSW": str(int(float(r.get("IDSW", "0")))),
                "FP": str(int(float(r.get("CLR_FP", "0")))),
                "FN": str(int(float(r.get("CLR_FN", "0")))),
                "Frag": str(int(float(r.get("Frag", "0")))),
            })


def run_official_trackeval(output_dir: Path, trackeval_root: Optional[Path]) -> None:
    trackeval = _import_official_trackeval(trackeval_root)
    data_root = output_dir / "trackeval_data"
    raw_out = output_dir / "trackeval_raw_outputs"
    raw_out.mkdir(parents=True, exist_ok=True)

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update({
        "USE_PARALLEL": False,
        "NUM_PARALLEL_CORES": 4,
        "BREAK_ON_ERROR": True,
        "PRINT_RESULTS": True,
        "PRINT_ONLY_COMBINED": False,
        "OUTPUT_SUMMARY": True,
        "OUTPUT_DETAILED": True,
        "PLOT_CURVES": False,
    })
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update({
        "GT_FOLDER": str(data_root / "gt" / "mot_challenge"),
        "TRACKERS_FOLDER": str(data_root / "trackers" / "mot_challenge"),
        "OUTPUT_FOLDER": str(raw_out),
        "TRACKERS_TO_EVAL": ["AMT_L48"],
        "CLASSES_TO_EVAL": ["pedestrian"],
        "BENCHMARK": "MFT25",
        "SPLIT_TO_EVAL": "val",
        "INPUT_AS_ZIP": False,
        "DO_PREPROC": False,
        "TRACKER_SUB_FOLDER": "data",
        "OUTPUT_SUB_FOLDER": "",
        "SEQMAP_FILE": str(data_root / "gt" / "mot_challenge" / "seqmaps" / "MFT25-val.txt"),
        "SKIP_SPLIT_FOL": False,
    })
    metrics_config = {"METRICS": ["HOTA", "CLEAR", "Identity"], "THRESHOLD": 0.5}
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [
        trackeval.metrics.HOTA(metrics_config),
        trackeval.metrics.CLEAR(metrics_config),
        trackeval.metrics.Identity(metrics_config),
    ]
    evaluator.evaluate(dataset_list, metrics_list)
    _write_summary_csv(output_dir)
    (output_dir / "evaluation_protocol.txt").write_text(
        "Evaluator: official TrackEval. DO_PREPROC=False is used because MFT25 is a single-class fish dataset.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate AMT-L48 with official TrackEval.")
    parser.add_argument("--data_root", type=Path, default=Path(DATA_ROOT))
    parser.add_argument("--checkpoint", type=Path, default=Path(REID_CKPT))
    parser.add_argument("--output_dir", type=Path, default=Path("results/amt_l48_trackeval"))
    parser.add_argument("--trackeval_root", type=Path, default=None)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--amp", type=int, default=1, choices=[0, 1])
    parser.add_argument("--emb_batch", type=int, default=EMB_BATCH)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--skip_export", action="store_true", help="Use existing MOT files in output_dir.")
    args = parser.parse_args()

    data_root = _resolve_path(args.data_root)
    checkpoint = _resolve_path(args.checkpoint)
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_export:
        export_amt_l48_mot(
            data_root=data_root,
            checkpoint=checkpoint,
            output_dir=output_dir,
            device=args.device,
            amp=bool(args.amp),
            emb_batch=int(args.emb_batch),
            num_workers=int(args.num_workers),
        )
    run_official_trackeval(output_dir=output_dir, trackeval_root=args.trackeval_root)


if __name__ == "__main__":
    main()
