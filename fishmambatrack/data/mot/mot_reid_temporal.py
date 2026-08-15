"""
fishmambatrack.data.mot.mot_reid_temporal

Tracklet-level ReID dataset (sequence of crops per identity).
"""

from __future__ import annotations

import argparse
import json
import pickle
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

try:
    import torch
    from torch.utils.data import Dataset
except Exception as e:
    raise RuntimeError("PyTorch is required for mot_reid_temporal.py") from e

from PIL import Image

from .mot_seq import MOTSequence
from .mot_utils import discover_sequence_dirs
from .image_ops import crop_tlwh, default_transform


@dataclass
class ReIDSeqItem:
    track_key: Tuple[str, int]
    pid: int
    track_id: int
    start: int
    length: int
    frame_stride: int


def _split_segments(frames: List[int], *, max_gap: int) -> List[Tuple[int, int]]:
    if not frames:
        return []
    segs: List[Tuple[int, int]] = []
    s = 0
    for i in range(1, len(frames)):
        if int(frames[i] - frames[i - 1]) > int(max_gap):
            segs.append((s, i))
            s = i
    segs.append((s, len(frames)))
    return segs


def _resize_with_pad(
    img: Image.Image,
    out_size: Tuple[int, int],
    *,
    pad_color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    out_h, out_w = int(out_size[0]), int(out_size[1])
    w, h = img.size
    if w <= 0 or h <= 0:
        return Image.new("RGB", (out_w, out_h), pad_color)
    scale = min(out_w / float(w), out_h / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = img.resize((new_w, new_h), resample=Image.BILINEAR)
    canvas = Image.new("RGB", (out_w, out_h), pad_color)
    left = (out_w - new_w) // 2
    top = (out_h - new_h) // 2
    canvas.paste(resized, (left, top))
    return canvas


def _min_len_for_seq(seq_len: int, frame_stride: int) -> int:
    return 1 + (int(seq_len) - 1) * int(frame_stride)


class MOTReIDTrackletDataset(Dataset):
    """
    Returns:
      x_seq: (T,3,H,W), pid
    """

    def __init__(
        self,
        root: Union[str, Path],
        *,
        seq_glob: str = "BT-*",
        gt_name: str = "gt_train_half.txt",
        full_gt_name: str = "gt.txt",
        img_dir_name: str = "img1",
        seq_len: int = 12,
        frame_stride: int = 1,
        seq_stride: int = 2,
        max_frame_gap: int = 1,
        # crop/transform
        crop_pad_ratio: float = 0.10,
        out_size: Tuple[int, int] = (256, 128),  # (H,W)
        normalize: bool = True,
        transform=None,
        # crop cache
        crop_cache_dir: Optional[Union[str, Path]] = None,
        crop_cache_format: str = "jpg",
        crop_cache_quality: int = 90,
        crop_cache_strict: bool = True,
        crop_cache_lazy: bool = True,
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
        self.seq_glob = str(seq_glob)
        self.gt_name = str(gt_name)
        self.full_gt_name = str(full_gt_name)
        self.img_dir_name = str(img_dir_name)

        self.seq_len = int(seq_len)
        self.frame_stride = int(frame_stride)
        self.seq_stride = int(seq_stride)
        self.max_frame_gap = int(max_frame_gap)
        if self.seq_len <= 0 or self.frame_stride <= 0 or self.seq_stride <= 0:
            raise ValueError("seq_len, frame_stride, and seq_stride must be positive.")
        if self.max_frame_gap < 1:
            raise ValueError("max_frame_gap must be at least 1.")

        self.crop_pad_ratio = float(crop_pad_ratio)
        self.out_size = tuple(out_size)
        self.normalize = bool(normalize)
        self.transform = transform
        self.return_meta = bool(return_meta)

        self.crop_cache_dir = (
            Path(crop_cache_dir) if crop_cache_dir is not None else None
        )
        self.crop_cache_format = str(crop_cache_format).lower().strip()
        self.crop_cache_quality = int(crop_cache_quality)
        self.crop_cache_strict = bool(crop_cache_strict)
        self.crop_cache_lazy = bool(crop_cache_lazy)
        if self.crop_cache_dir is not None:
            self.crop_cache_dir.mkdir(parents=True, exist_ok=True)
            self._check_crop_cache_meta()

        self.items: List[ReIDSeqItem] = []
        self.pid_to_indices: Dict[int, List[int]] = {}
        self.pid_map: Dict[Tuple[str, int], int] = {}
        self.track_map: Dict[
            Tuple[str, int],
            List[Tuple[int, int, str, Tuple[float, float, float, float]]],
        ] = {}

        self.cache_path = Path(cache_path) if cache_path is not None else None
        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        if (
            self.cache_path is not None
            and self.cache_path.exists()
            and (not rebuild_cache)
        ):
            self._load_cache(self.cache_path)
        else:
            self._build(skip_missing_gt=skip_missing_gt, limit_seqs=limit_seqs)
            if self.cache_path is not None:
                self._save_cache(self.cache_path)

        self.pid_to_indices = {}
        for i, it in enumerate(self.items):
            self.pid_to_indices.setdefault(it.pid, []).append(i)
        self.num_pids = len(self.pid_to_indices)

    def _save_cache(self, path: Path) -> None:
        payload = {
            "items": self.items,
            "pid_map": self.pid_map,
            "track_map": self.track_map,
            "meta": {
                "root": str(self.root),
                "seq_glob": self.seq_glob,
                "gt_name": self.gt_name,
                "full_gt_name": self.full_gt_name,
                "img_dir_name": self.img_dir_name,
                "seq_len": self.seq_len,
                "frame_stride": self.frame_stride,
                "seq_stride": self.seq_stride,
                "max_frame_gap": self.max_frame_gap,
                "crop_pad_ratio": self.crop_pad_ratio,
            },
        }
        with path.open("wb") as f:
            pickle.dump(payload, f)
        print(f"[MOTReIDTrackletDataset] Cache saved: {path} (items={len(self.items)})")

    def _load_cache(self, path: Path) -> None:
        class _SeqUnpickler(pickle.Unpickler):
            def find_class(self, module: str, name: str):
                if name == "ReIDSeqItem" and module in {
                    "__main__",
                    "mot_reid_temporal",
                }:
                    return ReIDSeqItem
                return super().find_class(module, name)

        with path.open("rb") as f:
            payload = _SeqUnpickler(f).load()

        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        if isinstance(meta, dict):
            mismatches = []
            expected = {
                "root": str(self.root),
                "seq_glob": str(self.seq_glob),
                "gt_name": str(self.gt_name),
                "full_gt_name": str(self.full_gt_name),
                "img_dir_name": str(self.img_dir_name),
                "seq_len": int(self.seq_len),
                "frame_stride": int(self.frame_stride),
                "seq_stride": int(self.seq_stride),
                "max_frame_gap": int(self.max_frame_gap),
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
        self.track_map = payload.get("track_map", {})
        print(
            f"[MOTReIDTrackletDataset] Cache loaded: {path} (items={len(self.items)})"
        )

    def _crop_cache_meta_path(self) -> Optional[Path]:
        if self.crop_cache_dir is None:
            return None
        return self.crop_cache_dir / "meta.json"

    def _check_crop_cache_meta(self) -> None:
        meta_path = self._crop_cache_meta_path()
        if meta_path is None:
            return
        expected = {
            "root": str(self.root.resolve()),
            "seq_glob": str(self.seq_glob),
            "gt_name": str(self.gt_name),
            "full_gt_name": str(self.full_gt_name),
            "img_dir_name": str(self.img_dir_name),
            "crop_pad_ratio": float(self.crop_pad_ratio),
            "out_size": list(self.out_size),
            "format": self.crop_cache_format,
            "resize_mode": "pad",
        }
        if not meta_path.exists():
            existing_files = [
                path
                for path in self.crop_cache_dir.rglob("*")
                if path.is_file() and path != meta_path
            ]
            if existing_files and self.crop_cache_strict:
                raise RuntimeError(
                    f"Crop cache contains files but has no metadata: {self.crop_cache_dir}"
                )
            temporary = meta_path.with_name(f".{meta_path.name}.{uuid.uuid4().hex}.tmp")
            temporary.write_text(
                json.dumps(expected, indent=2, sort_keys=True), encoding="utf-8"
            )
            temporary.replace(meta_path)
            return
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        mismatches = []
        for k, v in expected.items():
            if meta.get(k, None) != v:
                mismatches.append((k, meta.get(k, None), v))
        if mismatches and self.crop_cache_strict:
            lines = [
                f"Crop cache meta mismatch for {meta_path} (please rebuild cache):"
            ]
            for k, got, exp in mismatches:
                lines.append(f"  - {k}: cache={got!r} expected={exp!r}")
            raise RuntimeError("\n".join(lines))

    def _crop_cache_path(
        self, *, seq_name: str, global_frame: int, track_id: int
    ) -> Path:
        if self.crop_cache_dir is None:
            raise RuntimeError("crop_cache_dir is not set")
        ext = self.crop_cache_format
        return (
            self.crop_cache_dir
            / seq_name
            / f"{int(global_frame):06d}_tid{int(track_id)}.{ext}"
        )

    def _save_crop_cache(self, crop: Image.Image, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ext = self.crop_cache_format
        # DataLoader workers can request the same crop concurrently; use a
        # unique temp name so atomic replace does not race on a shared tmp file.
        tmp_path = path.with_name(f"{path.stem}.tmp.{uuid.uuid4().hex}.{ext}")
        if ext == "jpg":
            crop.save(tmp_path, quality=int(self.crop_cache_quality))
        else:
            crop.save(tmp_path)
        tmp_path.replace(path)

    def _build(self, *, skip_missing_gt: bool, limit_seqs: Optional[int]) -> None:
        seq_dirs = discover_sequence_dirs(
            self.root, seq_glob=self.seq_glob, img_dir_name=self.img_dir_name, sort=True
        )
        if limit_seqs is not None:
            seq_dirs = seq_dirs[: int(limit_seqs)]
        if not seq_dirs:
            raise RuntimeError(
                f"No sequences found under {self.root} with glob={self.seq_glob}"
            )

        pid_map: Dict[Tuple[str, int], int] = {}
        next_pid = 0
        items: List[ReIDSeqItem] = []
        track_map: Dict[
            Tuple[str, int],
            List[Tuple[int, int, str, Tuple[float, float, float, float]]],
        ] = {}

        used = 0
        min_len = _min_len_for_seq(self.seq_len, self.frame_stride)

        for seq_dir in seq_dirs:
            gt_path = seq_dir / "gt" / self.gt_name
            if not gt_path.exists():
                if skip_missing_gt:
                    print(f"[MOTReIDTrackletDataset] Skip (missing GT): {gt_path}")
                    continue
                raise FileNotFoundError(gt_path)

            seq = MOTSequence(
                seq_dir,
                gt_relpath=f"gt/{self.gt_name}",
                full_gt_relpath=f"gt/{self.full_gt_name}",
                det_relpath=None,
                img_dir_name=self.img_dir_name,
                drop_ignored_gt=True,
            )

            track_to_list: Dict[
                int, List[Tuple[int, Tuple[float, float, float, float]]]
            ] = {}
            for f, recs in seq.gt_by_frame.items():
                for r in recs:
                    track_to_list.setdefault(r.track_id, []).append((f, r.tlwh))

            for track_id, lst in track_to_list.items():
                lst.sort(key=lambda t: t[0])
                if len(lst) < min_len:
                    continue

                key = (seq.seq_name, int(track_id))
                if key not in pid_map:
                    pid_map[key] = next_pid
                    next_pid += 1
                pid = pid_map[key]

                entries = []
                for f, tlwh in lst:
                    g = seq.local_to_global(f)
                    img_path = str(seq.get_image_path(f))
                    entries.append((int(f), int(g), img_path, tuple(map(float, tlwh))))
                track_map[key] = entries

                frames = [e[0] for e in entries]
                for s, e in _split_segments(frames, max_gap=self.max_frame_gap):
                    if (e - s) < min_len:
                        continue
                    end_start = e - min_len
                    for start in range(s, end_start + 1, int(self.seq_stride)):
                        items.append(
                            ReIDSeqItem(
                                track_key=key,
                                pid=int(pid),
                                track_id=int(track_id),
                                start=int(start),
                                length=int(self.seq_len),
                                frame_stride=int(self.frame_stride),
                            )
                        )

            used += 1

        if used == 0:
            raise RuntimeError(
                f"No usable sequences found (gt_name={self.gt_name}) under {self.root}"
            )

        items.sort(key=lambda it: (it.track_key[0], it.pid, it.start))
        self.items = items
        self.pid_map = pid_map
        self.track_map = track_map

        print(
            f"[MOTReIDTrackletDataset] Built items={len(self.items)} "
            f"pids={len({it.pid for it in items})} seqs={used} "
            f"seq_len={self.seq_len} frame_stride={self.frame_stride} "
            f"seq_stride={self.seq_stride}"
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        it = self.items[index]
        track = self.track_map[it.track_key]
        idxs = [it.start + i * it.frame_stride for i in range(it.length)]
        if idxs[-1] >= len(track):
            raise IndexError(f"Sequence index out of range for track {it.track_key}")

        xs: List[torch.Tensor] = []
        metas = []
        for i, ti in enumerate(idxs):
            frame, global_frame, img_path, tlwh = track[ti]
            crop = None
            if self.crop_cache_dir is not None:
                cache_path = self._crop_cache_path(
                    seq_name=it.track_key[0],
                    global_frame=global_frame,
                    track_id=it.track_id,
                )
                if cache_path.exists():
                    with Image.open(cache_path) as _im:
                        crop = _im.convert("RGB")
            if crop is None:
                with Image.open(img_path) as _im:
                    img = _im.convert("RGB")

                crop = crop_tlwh(img, tlwh, pad_ratio=self.crop_pad_ratio)
                crop = _resize_with_pad(crop, self.out_size)
                if self.crop_cache_dir is not None and self.crop_cache_lazy:
                    cache_path = self._crop_cache_path(
                        seq_name=it.track_key[0],
                        global_frame=global_frame,
                        track_id=it.track_id,
                    )
                    self._save_crop_cache(crop, cache_path)

            if self.transform is None:
                x = default_transform(
                    crop, size=self.out_size, normalize=self.normalize
                )
            else:
                x = self.transform(crop)

            xs.append(x)
            if self.return_meta:
                metas.append(
                    {
                        "seq_name": it.track_key[0],
                        "track_id": it.track_id,
                        "frame": frame,
                        "global_frame": global_frame,
                        "img_path": img_path,
                        "tlwh": tlwh,
                    }
                )

        x_seq = torch.stack(xs, dim=0)
        pid = int(it.pid)
        if self.return_meta:
            return x_seq, pid, metas
        return x_seq, pid


def _main() -> None:
    ap = argparse.ArgumentParser("Build and inspect MOTReIDTrackletDataset.")
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--seq_glob", type=str, default="BT-*")
    ap.add_argument("--gt_name", type=str, default="gt_train_half.txt")
    ap.add_argument("--full_gt_name", type=str, default="gt.txt")
    ap.add_argument("--seq_len", type=int, default=12)
    ap.add_argument("--frame_stride", type=int, default=1)
    ap.add_argument("--seq_stride", type=int, default=2)
    ap.add_argument("--max_gap", type=int, default=1)
    ap.add_argument("--limit_seqs", type=int, default=None)
    ap.add_argument("--cache", type=str, default=None)
    ap.add_argument("--rebuild_cache", action="store_true")
    args = ap.parse_args()

    ds = MOTReIDTrackletDataset(
        args.root,
        seq_glob=args.seq_glob,
        gt_name=args.gt_name,
        full_gt_name=args.full_gt_name,
        seq_len=args.seq_len,
        frame_stride=args.frame_stride,
        seq_stride=args.seq_stride,
        max_frame_gap=args.max_gap,
        cache_path=args.cache,
        rebuild_cache=args.rebuild_cache,
        return_meta=True,
        limit_seqs=args.limit_seqs,
    )
    print(f"Dataset len={len(ds)} num_pids={ds.num_pids} seq_len={ds.seq_len}")
    for i in range(min(3, len(ds))):
        x, pid, meta = ds[i]
        print(
            f"[{i}] pid={pid} x.shape={tuple(x.shape)} frames={[m['frame'] for m in meta]}"
        )


if __name__ == "__main__":
    _main()
