"""
fishmambatrack.data.mot.mot_reid

Build ReID training samples from MOT-format GT (bbox + id).
No extra labels required.

Key output per sample:
- cropped image tensor
- global person id (pid) (unique across sequences)
- axis pseudo-label: (cos(2theta), sin(2theta)) derived from velocity direction
- axis weight r in [0,1] based on speed magnitude (optionally normalized by bbox size)

This module is designed for "get it running first":
- Use GT split files (gt_train_half.txt / gt_val_half.txt) for supervision
- Infer split->global frame offset using full GT (gt.txt) if needed
"""

from __future__ import annotations

import argparse
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except Exception as e:
    raise RuntimeError("PyTorch is required for mot_reid.py") from e

from PIL import Image
from PIL import ImageEnhance, ImageFilter

from .mot_seq import MOTSequence
from .mot_utils import discover_sequence_dirs


# -----------------------------
# Data structure
# -----------------------------

@dataclass
class ReIDItem:
    seq_name: str
    pid: int              # global unique id across sequences
    track_id: int         # original MOT id within the sequence
    frame: int            # local frame index in the chosen gt file
    global_frame: int     # global frame index used for image files
    img_path: str
    tlwh: Tuple[float, float, float, float]

    # Velocity-guided axis pseudo label
    axis: Tuple[float, float]        # (cos2theta, sin2theta)
    axis_weight: float               # r in [0,1]
    speed: float                     # speed after smoothing (px/frame or normalized)
    vxy: Tuple[float, float]         # smoothed velocity (vx, vy)


# -----------------------------
# Helpers: image crop + tensor
# -----------------------------

def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def crop_tlwh(
    img: Image.Image,
    tlwh: Tuple[float, float, float, float],
    *,
    pad_ratio: float = 0.10,
) -> Image.Image:
    """Crop bbox with padding and boundary clipping."""
    W, H = img.size
    x, y, w, h = tlwh
    # padding proportional to box size
    padx = w * pad_ratio
    pady = h * pad_ratio

    x1 = int(math.floor(x - padx))
    y1 = int(math.floor(y - pady))
    x2 = int(math.ceil(x + w + padx))
    y2 = int(math.ceil(y + h + pady))

    x1 = int(_clip(x1, 0, W - 1))
    y1 = int(_clip(y1, 0, H - 1))
    x2 = int(_clip(x2, 1, W))
    y2 = int(_clip(y2, 1, H))

    # Ensure valid crop
    if x2 <= x1 + 1:
        x2 = min(W, x1 + 2)
    if y2 <= y1 + 1:
        y2 = min(H, y1 + 2)

    return img.crop((x1, y1, x2, y2))


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL RGB image to float tensor in [0,1], shape (3,H,W)."""
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    # RGB
    arr = arr / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def normalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    """In-place-ish normalize (ImageNet mean/std)."""
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device)[:, None, None]
    return (x - mean) / std


def default_transform(
    img: Image.Image,
    *,
    size: Tuple[int, int] = (256, 128),  # (H, W) typical ReID
    normalize: bool = True,
) -> torch.Tensor:
    img = img.convert("RGB")
    img = img.resize((size[1], size[0]), resample=Image.BILINEAR)
    x = pil_to_tensor(img)  # (3,H,W)
    if normalize:
        x = normalize_imagenet(x)
    return x


def _apply_color_jitter(
    img: Image.Image,
    *,
    strength: float,
) -> Image.Image:
    """Lightweight color jitter without requiring torchvision."""
    s = float(strength)
    if s <= 0:
        return img

    # brightness/contrast/saturation factors in [1-s, 1+s]
    b = 1.0 + random.uniform(-s, s)
    c = 1.0 + random.uniform(-s, s)
    sat = 1.0 + random.uniform(-s, s)

    img = ImageEnhance.Brightness(img).enhance(b)
    img = ImageEnhance.Contrast(img).enhance(c)
    img = ImageEnhance.Color(img).enhance(sat)
    return img


def _random_erasing_(
    x: torch.Tensor,
    *,
    p: float,
    area: Tuple[float, float] = (0.02, 0.20),
    aspect: Tuple[float, float] = (0.3, 3.3),
    value: float = 0.0,
) -> torch.Tensor:
    """Apply Random Erasing on (C,H,W) tensor (in-place-ish)."""
    if float(p) <= 0 or random.random() >= float(p):
        return x

    if x.ndim != 3:
        return x

    C, H, W = x.shape
    if H < 2 or W < 2:
        return x

    a0, a1 = float(area[0]), float(area[1])
    r0, r1 = float(aspect[0]), float(aspect[1])
    a0 = max(0.0, min(1.0, a0))
    a1 = max(0.0, min(1.0, a1))
    if a1 <= a0 or r1 <= r0:
        return x

    img_area = H * W
    for _ in range(10):
        target_area = random.uniform(a0, a1) * img_area
        aspect_ratio = random.uniform(r0, r1)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / max(1e-9, aspect_ratio))))
        if h <= 0 or w <= 0 or h >= H or w >= W:
            continue

        top = random.randint(0, H - h)
        left = random.randint(0, W - w)
        x[:, top : top + h, left : left + w] = float(value)
        break
    return x


# -----------------------------
# Velocity -> axis pseudo label
# -----------------------------

def _bbox_center(tlwh: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x, y, w, h = tlwh
    return (x + 0.5 * w, y + 0.5 * h)


def _size_ref(
    tlwh: Tuple[float, float, float, float],
    mode: str,
) -> float:
    _, _, w, h = tlwh
    if mode == "none":
        return 1.0
    if mode == "sqrt_area":
        return math.sqrt(max(1.0, w * h))
    if mode == "max_side":
        return max(1.0, max(w, h))
    raise ValueError(f"Unknown vel_norm_mode='{mode}' (use none/sqrt_area/max_side)")


def _axis_from_v(vx: float, vy: float) -> Tuple[float, float]:
    # For axis (no head-tail), use 2*theta representation: (cos2θ, sin2θ)
    if abs(vx) < 1e-9 and abs(vy) < 1e-9:
        # arbitrary; weight should be ~0 in this case
        return (1.0, 0.0)
    theta = math.atan2(vy, vx)
    return (math.cos(2.0 * theta), math.sin(2.0 * theta))


def _reliability_from_speed(speed: float, s0: float, s1: float) -> float:
    """Clamp linear ramp [s0,s1] -> [0,1]."""
    if s1 <= s0:
        return 1.0 if speed > s0 else 0.0
    r = (speed - s0) / (s1 - s0)
    return float(_clip(r, 0.0, 1.0))


def _regress_velocity(hist: List[Tuple[int, Tuple[float, float]]]) -> Optional[Tuple[float, float]]:
    """Estimate velocity by linear regression on recent centers."""
    if len(hist) < 2:
        return None
    t = np.array([float(h[0]) for h in hist], dtype=np.float32)
    x = np.array([float(h[1][0]) for h in hist], dtype=np.float32)
    y = np.array([float(h[1][1]) for h in hist], dtype=np.float32)
    t_mean = float(t.mean())
    dt = t - t_mean
    denom = float((dt * dt).sum())
    if denom <= 1e-6:
        return None
    vx = float((dt * (x - float(x.mean()))).sum() / denom)
    vy = float((dt * (y - float(y.mean()))).sum() / denom)
    return (vx, vy)


# -----------------------------
# Dataset builder
# -----------------------------

def build_reid_items_from_sequence(
    seq: MOTSequence,
    *,
    pid_map: Dict[Tuple[str, int], int],
    next_pid: int,
    vel_smooth: float = 0.8,
    vel_mode: str = "ema",  # ema | regress
    vel_window: int = 5,
    vel_norm_mode: str = "sqrt_area",
    r_s0: float = 0.01,
    r_s1: float = 0.05,
    pad_ratio: float = 0.10,
) -> Tuple[List[ReIDItem], int]:
    """
    Build ReIDItem list for a single sequence using seq.gt_by_frame.

    Returns:
      items, updated next_pid
    """
    # Collect per-track time-ordered records
    track_to_list: Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]] = {}
    for f, recs in seq.gt_by_frame.items():
        for r in recs:
            track_to_list.setdefault(r.track_id, []).append((f, r.tlwh))

    items: List[ReIDItem] = []

    for track_id, lst in track_to_list.items():
        lst.sort(key=lambda t: t[0])

        key = (seq.seq_name, int(track_id))
        if key not in pid_map:
            pid_map[key] = next_pid
            next_pid += 1
        pid = pid_map[key]

        prev_center: Optional[Tuple[float, float]] = None
        prev_frame: Optional[int] = None
        vbarx, vbary = 0.0, 0.0
        hist: List[Tuple[int, Tuple[float, float]]] = []
        vel_mode = str(vel_mode).lower().strip()
        vel_window = max(2, int(vel_window))

        for f, tlwh in lst:
            img_path = str(seq.get_image_path(f))
            g = seq.local_to_global(f)

            # velocity estimate
            c = _bbox_center(tlwh)

            if prev_center is None or prev_frame is None:
                # first observation: no reliable velocity
                vbarx, vbary = 0.0, 0.0
                speed = 0.0
                axis = _axis_from_v(vbarx, vbary)
                r = 0.0
            else:
                dt = max(1, int(f - prev_frame))
                vx = (c[0] - prev_center[0]) / dt
                vy = (c[1] - prev_center[1]) / dt

                hist.append((int(f), c))
                if len(hist) > vel_window:
                    hist = hist[-vel_window:]
                if vel_mode == "regress":
                    v_reg = _regress_velocity(hist)
                    if v_reg is not None:
                        vx, vy = v_reg

                # EMA smoothing
                vbarx = vel_smooth * vbarx + (1.0 - vel_smooth) * vx
                vbary = vel_smooth * vbary + (1.0 - vel_smooth) * vy

                # speed (optionally normalized by bbox size)
                denom = _size_ref(tlwh, vel_norm_mode)
                # If mode == "none", denom == 1
                speed = math.sqrt(vbarx * vbarx + vbary * vbary) / max(1e-6, denom)

                # reliability in [0,1]
                r = _reliability_from_speed(speed, r_s0, r_s1)

                # If gaps exist (dt>1), reduce reliability slightly
                if dt > 1:
                    r *= min(1.0, 1.0 / float(dt))

                axis = _axis_from_v(vbarx, vbary)

            item = ReIDItem(
                seq_name=seq.seq_name,
                pid=pid,
                track_id=int(track_id),
                frame=int(f),
                global_frame=int(g),
                img_path=img_path,
                tlwh=tuple(map(float, tlwh)),
                axis=axis,
                axis_weight=float(r),
                speed=float(speed),
                vxy=(float(vbarx), float(vbary)),
            )
            items.append(item)

        prev_center = c
        prev_frame = f

    return items, next_pid


def compute_speed_stats(items: Sequence[ReIDItem]) -> Dict[str, float]:
    if not items:
        return {"n": 0.0}
    speeds = np.array([it.speed for it in items], dtype=np.float32)
    return {
        "n": float(len(items)),
        "speed_min": float(np.min(speeds)),
        "speed_p50": float(np.percentile(speeds, 50)),
        "speed_p90": float(np.percentile(speeds, 90)),
        "speed_p95": float(np.percentile(speeds, 95)),
        "speed_max": float(np.max(speeds)),
        "speed_mean": float(np.mean(speeds)),
    }


# -----------------------------
# PyTorch Dataset
# -----------------------------

class MOTReIDDataset(Dataset):
    """
    Dataset returning:
      img_tensor, pid, axis_tensor(2,), axis_weight(float)

    By default, img is cropped from GT bbox and resized to (H,W) = (256,128).
    """

    def __init__(
        self,
        root: Union[str, Path],
        *,
        seq_glob: str = "BT-*",
        gt_name: str = "gt_train_half.txt",
        full_gt_name: str = "gt.txt",
        img_dir_name: str = "img1",
        # velocity pseudo label
        vel_smooth: float = 0.8,
        vel_mode: str = "ema",  # ema | regress
        vel_window: int = 5,
        vel_norm_mode: str = "sqrt_area",  # none/sqrt_area/max_side
        r_s0: float = 0.01,
        r_s1: float = 0.05,
        # crop/transform
        crop_pad_ratio: float = 0.10,
        out_size: Tuple[int, int] = (256, 128),  # (H,W)
        normalize: bool = True,
        transform=None,
        # augmentation (train only)
        augment: bool = False,
        bbox_jitter_prob: float = 0.80,
        bbox_jitter_center: float = 0.05,
        bbox_jitter_scale: float = 0.15,
        pad_ratio_jitter: float = 0.50,
        color_jitter: float = 0.20,
        blur_prob: float = 0.05,
        blur_radius_max: float = 1.5,
        noise_std: float = 0.00,
        random_erasing_prob: float = 0.25,
        random_erasing_value: float = 0.0,
        # caching
        cache_path: Optional[Union[str, Path]] = None,
        rebuild_cache: bool = False,
        # return meta
        return_meta: bool = False,
        skip_missing_gt: bool = True,
        limit_seqs: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.seq_glob = seq_glob
        self.gt_name = gt_name
        self.full_gt_name = full_gt_name
        self.img_dir_name = img_dir_name

        self.vel_smooth = float(vel_smooth)
        self.vel_mode = str(vel_mode)
        self.vel_window = int(vel_window)
        self.vel_norm_mode = str(vel_norm_mode)
        self.r_s0 = float(r_s0)
        self.r_s1 = float(r_s1)

        self.crop_pad_ratio = float(crop_pad_ratio)
        self.out_size = tuple(out_size)
        self.normalize = bool(normalize)
        self.transform = transform  # if None, use default_transform
        self.return_meta = bool(return_meta)

        # augmentation knobs
        self.augment = bool(augment)
        self.bbox_jitter_prob = float(bbox_jitter_prob)
        self.bbox_jitter_center = float(bbox_jitter_center)
        self.bbox_jitter_scale = float(bbox_jitter_scale)
        self.pad_ratio_jitter = float(pad_ratio_jitter)
        self.color_jitter = float(color_jitter)
        self.blur_prob = float(blur_prob)
        self.blur_radius_max = float(blur_radius_max)
        self.noise_std = float(noise_std)
        self.random_erasing_prob = float(random_erasing_prob)
        self.random_erasing_value = float(random_erasing_value)

        self.items: List[ReIDItem] = []
        self.pid_to_indices: Dict[int, List[int]] = {}
        self.pid_map: Dict[Tuple[str, int], int] = {}

        self.cache_path = Path(cache_path) if cache_path is not None else None
        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        if self.cache_path is not None and self.cache_path.exists() and (not rebuild_cache):
            self._load_cache(self.cache_path)
        else:
            self._build(skip_missing_gt=skip_missing_gt, limit_seqs=limit_seqs)
            if self.cache_path is not None:
                self._save_cache(self.cache_path)

        # Build pid_to_indices for samplers
        self.pid_to_indices = {}
        for i, it in enumerate(self.items):
            self.pid_to_indices.setdefault(it.pid, []).append(i)

        self.num_pids = len(self.pid_to_indices)

    def _save_cache(self, path: Path) -> None:
        payload = {
            "items": self.items,
            "pid_map": self.pid_map,
            "meta": {
                "root": str(self.root),
                "seq_glob": self.seq_glob,
                "gt_name": self.gt_name,
                "full_gt_name": self.full_gt_name,
                "img_dir_name": self.img_dir_name,
                "vel_smooth": self.vel_smooth,
                "vel_mode": self.vel_mode,
                "vel_window": self.vel_window,
                "vel_norm_mode": self.vel_norm_mode,
                "r_s0": self.r_s0,
                "r_s1": self.r_s1,
                "crop_pad_ratio": self.crop_pad_ratio,
            },
        }
        with path.open("wb") as f:
            pickle.dump(payload, f)
        print(f"[MOTReIDDataset] Cache saved: {path} (items={len(self.items)})")

    def _load_cache(self, path: Path) -> None:
        class _ReIDUnpickler(pickle.Unpickler):
            """Handle older caches pickled from __main__ or short module names."""
            def find_class(self, module: str, name: str):
                if name == "ReIDItem" and module in {"__main__", "mot_reid"}:
                    return ReIDItem
                return super().find_class(module, name)

        with path.open("rb") as f:
            try:
                payload = _ReIDUnpickler(f).load()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load cache {path}. "
                    "Delete the cache or run with rebuild_cache=1 to regenerate."
                ) from e

        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        if isinstance(meta, dict):
            mismatches = []
            expected = {
                "root": str(self.root),
                "seq_glob": str(self.seq_glob),
                "gt_name": str(self.gt_name),
                "full_gt_name": str(self.full_gt_name),
                "img_dir_name": str(self.img_dir_name),
            }
            for k, v in expected.items():
                mv = meta.get(k, None)
                if mv != v:
                    mismatches.append((k, mv, v))

            if mismatches:
                lines = [f"Cache meta mismatch for {path} (please rebuild cache):"]
                for k, got, exp in mismatches:
                    lines.append(f"  - {k}: cache={got!r} expected={exp!r}")
                raise RuntimeError("\n".join(lines))

        self.items = payload["items"]
        self.pid_map = payload.get("pid_map", {})
        print(f"[MOTReIDDataset] Cache loaded: {path} (items={len(self.items)})")

    def _build(self, *, skip_missing_gt: bool, limit_seqs: Optional[int]) -> None:
        seq_dirs = discover_sequence_dirs(self.root, seq_glob=self.seq_glob, img_dir_name=self.img_dir_name, sort=True)
        if limit_seqs is not None:
            seq_dirs = seq_dirs[: int(limit_seqs)]

        if not seq_dirs:
            raise RuntimeError(f"No sequences found under {self.root} with glob={self.seq_glob}")

        pid_map: Dict[Tuple[str, int], int] = {}
        next_pid = 0
        all_items: List[ReIDItem] = []

        used = 0
        for seq_dir in seq_dirs:
            gt_path = seq_dir / "gt" / self.gt_name
            full_gt_path = seq_dir / "gt" / self.full_gt_name
            if not gt_path.exists():
                if skip_missing_gt:
                    print(f"[MOTReIDDataset] Skip (missing GT): {gt_path}")
                    continue
                raise FileNotFoundError(gt_path)

            # Build MOTSequence view (no det needed)
            seq = MOTSequence(
                seq_dir,
                gt_relpath=f"gt/{self.gt_name}",
                full_gt_relpath=f"gt/{self.full_gt_name}",
                det_relpath=None,
                img_dir_name=self.img_dir_name,
                drop_ignored_gt=True,
            )

            items, next_pid = build_reid_items_from_sequence(
                seq,
                pid_map=pid_map,
                next_pid=next_pid,
                vel_smooth=self.vel_smooth,
                vel_mode=self.vel_mode,
                vel_window=self.vel_window,
                vel_norm_mode=self.vel_norm_mode,
                r_s0=self.r_s0,
                r_s1=self.r_s1,
                pad_ratio=self.crop_pad_ratio,
            )
            all_items.extend(items)
            used += 1

        if used == 0:
            raise RuntimeError(f"No usable sequences found (gt_name={self.gt_name}) under {self.root}")

        # Deterministic ordering: by (seq, pid, global_frame)
        all_items.sort(key=lambda it: (it.seq_name, it.pid, it.global_frame))
        self.items = all_items
        self.pid_map = pid_map

        stats = compute_speed_stats(self.items)
        print(f"[MOTReIDDataset] Built items={len(self.items)} pids={len(set([it.pid for it in self.items]))} "
              f"seqs={used} vel_norm_mode={self.vel_norm_mode} r_s0={self.r_s0} r_s1={self.r_s1}")
        print(f"[MOTReIDDataset] Speed stats: {stats}")

    def __len__(self) -> int:
        return len(self.items)

    def _maybe_jitter_tlwh(self, tlwh: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        if (not self.augment) or (self.bbox_jitter_prob <= 0) or (random.random() >= self.bbox_jitter_prob):
            return tlwh

        x, y, w, h = map(float, tlwh)
        if w <= 1.0 or h <= 1.0:
            return tlwh

        cx = x + 0.5 * w
        cy = y + 0.5 * h

        # Center shift (simulate detection noise)
        dx = random.uniform(-1.0, 1.0) * self.bbox_jitter_center * w
        dy = random.uniform(-1.0, 1.0) * self.bbox_jitter_center * h

        # Scale jitter
        sw = math.exp(random.uniform(-self.bbox_jitter_scale, self.bbox_jitter_scale))
        sh = math.exp(random.uniform(-self.bbox_jitter_scale, self.bbox_jitter_scale))
        w2 = max(2.0, w * sw)
        h2 = max(2.0, h * sh)

        cx2 = cx + dx
        cy2 = cy + dy
        x2 = cx2 - 0.5 * w2
        y2 = cy2 - 0.5 * h2
        return (x2, y2, w2, h2)

    def _maybe_jitter_pad_ratio(self, pad_ratio: float) -> float:
        if (not self.augment) or self.pad_ratio_jitter <= 0:
            return float(pad_ratio)
        j = float(self.pad_ratio_jitter)
        lo = max(0.0, 1.0 - j)
        hi = 1.0 + j
        return float(pad_ratio) * random.uniform(lo, hi)

    def __getitem__(self, index: int):
        it = self.items[index]
        img = Image.open(it.img_path).convert("RGB")

        tlwh = self._maybe_jitter_tlwh(it.tlwh)
        pad_ratio = self._maybe_jitter_pad_ratio(self.crop_pad_ratio)
        crop = crop_tlwh(img, tlwh, pad_ratio=pad_ratio)

        if self.augment:
            crop = _apply_color_jitter(crop, strength=self.color_jitter)
            if self.blur_prob > 0 and random.random() < self.blur_prob:
                r = random.uniform(0.1, max(0.1, self.blur_radius_max))
                crop = crop.filter(ImageFilter.GaussianBlur(radius=r))

        if self.transform is None:
            x = default_transform(crop, size=self.out_size, normalize=self.normalize)
        else:
            x = self.transform(crop)

        # additive noise in pre-normalized scale is tricky; here we assume normalized tensor
        if self.augment and self.noise_std > 0:
            x = x + torch.randn_like(x) * float(self.noise_std)

        if self.augment:
            x = _random_erasing_(x, p=self.random_erasing_prob, value=self.random_erasing_value)

        pid = int(it.pid)
        axis = torch.tensor(it.axis, dtype=torch.float32)
        axis_w = float(it.axis_weight)

        if self.return_meta:
            meta = {
                "seq_name": it.seq_name,
                "track_id": it.track_id,
                "frame": it.frame,
                "global_frame": it.global_frame,
                "img_path": it.img_path,
                "tlwh": it.tlwh,
                "speed": it.speed,
                "vxy": it.vxy,
            }
            return x, pid, axis, axis_w, meta

        return x, pid, axis, axis_w


# -----------------------------
# Train/dev split helpers
# -----------------------------

def split_train_dev_tail_frames(
    items: Sequence[ReIDItem],
    *,
    dev_tail_frac: float = 0.20,
    ensure_dev_pids_in_train: bool = True,
) -> Tuple[List[int], List[int]]:
    """
    Split indices into (train_idx, dev_idx) by taking the last `dev_tail_frac`
    of frames (per-sequence) as dev.

    This is useful when you want a dev set without touching gt_val_half.txt.

    Notes:
      - Split is per sequence, based on unique frame ids in `it.frame`.
      - If `ensure_dev_pids_in_train=True`, dev samples whose pid never appears
        in the train split will be dropped (keeps CE dev meaningful).
    """
    if not (0.0 < float(dev_tail_frac) < 1.0):
        raise ValueError(f"dev_tail_frac must be in (0,1), got {dev_tail_frac}")

    # Collect unique frames per sequence (use local frame index in the gt file).
    seq_to_frames: Dict[str, set] = {}
    for it in items:
        seq_to_frames.setdefault(it.seq_name, set()).add(int(it.frame))

    # Determine dev frames per sequence.
    seq_to_dev_frames: Dict[str, set] = {}
    for seq_name, frames_set in seq_to_frames.items():
        frames = sorted(frames_set)
        if len(frames) < 2:
            seq_to_dev_frames[seq_name] = set()
            continue
        cut = int(math.floor((1.0 - float(dev_tail_frac)) * len(frames)))
        cut = max(1, min(len(frames) - 1, cut))
        seq_to_dev_frames[seq_name] = set(frames[cut:])

    train_idx: List[int] = []
    dev_idx: List[int] = []
    for i, it in enumerate(items):
        if int(it.frame) in seq_to_dev_frames.get(it.seq_name, set()):
            dev_idx.append(i)
        else:
            train_idx.append(i)

    if ensure_dev_pids_in_train:
        train_pids = {int(items[i].pid) for i in train_idx}
        dev_idx = [i for i in dev_idx if int(items[i].pid) in train_pids]

    return train_idx, dev_idx


# -----------------------------
# CLI: quick inspect
# -----------------------------

def _main() -> None:
    ap = argparse.ArgumentParser("Build and inspect MOTReIDDataset.")
    ap.add_argument("--root", type=str, required=True, help="Dataset root, e.g. data/MFT25-train")
    ap.add_argument("--seq_glob", type=str, default="BT-*")
    ap.add_argument("--gt_name", type=str, default="gt_train_half.txt")
    ap.add_argument("--full_gt_name", type=str, default="gt.txt")
    ap.add_argument("--limit_seqs", type=int, default=None)
    ap.add_argument("--cache", type=str, default=None)
    ap.add_argument("--rebuild_cache", action="store_true")
    ap.add_argument("--dump", type=int, default=0, help="Dump a few crops to outputs/reid_crops/")
    ap.add_argument("--dump_n", type=int, default=16)
    args = ap.parse_args()

    ds = MOTReIDDataset(
        args.root,
        seq_glob=args.seq_glob,
        gt_name=args.gt_name,
        full_gt_name=args.full_gt_name,
        cache_path=args.cache,
        rebuild_cache=args.rebuild_cache,
        return_meta=True,
        limit_seqs=args.limit_seqs,
    )
    print(f"Dataset len={len(ds)} num_pids={ds.num_pids}")

    # Show some samples
    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    for i in idxs[:5]:
        x, pid, axis, w, meta = ds[i]
        print(f"[{i}] pid={pid} axis={axis.tolist()} w={w:.3f} meta={{seq={meta['seq_name']}, "
              f"tid={meta['track_id']}, g={meta['global_frame']}, speed={meta['speed']:.4f}}} "
              f"x.shape={tuple(x.shape)}")

    if args.dump:
        out_dir = Path("outputs/reid_crops") / args.gt_name.replace(".txt", "")
        out_dir.mkdir(parents=True, exist_ok=True)
        for j, i in enumerate(idxs[: args.dump_n]):
            # re-open + crop for dumping (use raw crop, not normalized tensor)
            it = ds.items[i]
            img = Image.open(it.img_path).convert("RGB")
            crop = crop_tlwh(img, it.tlwh, pad_ratio=ds.crop_pad_ratio)
            out_path = out_dir / f"{j:03d}_pid{it.pid}_g{it.global_frame:06d}_tid{it.track_id}.jpg"
            crop.save(out_path)
        print(f"Dumped {min(args.dump_n, len(ds))} crops to: {out_dir}")


if __name__ == "__main__":
    _main()
