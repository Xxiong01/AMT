# FishMambaTrack

FishMambaTrack is a fish MOT research repo with a temporal ReID branch
(CNN + Mamba) and multiple tracking baselines.

This README documents the paper-default temporal ReID setup and common commands.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Data layout

Expected MOT-style layout (example sequence BT-001):

```
data/MFT25-train/BT-001/
  img1/000001.jpg ...
  gt/gt.txt
  gt/gt_train_half.txt
  gt/gt_val_half.txt
  det/det_yolox_ckpt.txt
```

## Temporal ReID (paper defaults)

Defaults are defined in `scripts/train_reid_mamba_temporal.py`.

Core settings:

- Crop: 128x256 (H x W), crop_pad=0.10
- Sliding window: L=8, frame_stride=1, seq_stride=3, max_gap=2
- Backbone: ResNet-18 (ImageNet pretrained)
- Temporal: 2-layer Mamba, emb_dim=256, dropout=0.10, pool=mean_last
- Loss: CE + Triplet (margin=0.3, batch-hard), weights W_CE=1.0, W_METRIC=1.0
- Optimizer: AdamW, lr=3e-4, weight_decay=1e-4
- Training: 100 epochs, batch_size=16, AMP=True

### Train

```bash
python scripts/train_reid_mamba_temporal.py
```

### Optional: pre-cache tracklet crops

This speeds up training and makes crops deterministic.

```bash
python scripts/precache_reid_tracklet_crops.py \
  --root data/MFT25-train \
  --gt_name gt_train_half.txt \
  --full_gt_name gt.txt \
  --out_dir outputs/reid_tracklet_crops_train \
  --pad 0.10 --out_h 128 --out_w 256

python scripts/precache_reid_tracklet_crops.py \
  --root data/MFT25-train \
  --gt_name gt_val_half.txt \
  --full_gt_name gt.txt \
  --out_dir outputs/reid_tracklet_crops_val \
  --pad 0.10 --out_h 128 --out_w 256
```

If you change seq_len / stride / gap / crop params, delete old caches:

```
outputs/cache_reid_tracklet_*.pkl
outputs/reid_tracklet_crops_*/
```

## Tracking evaluation (temporal ReID)

```bash
python scripts/tune_tracker_temporal.py \
  --cfg_names baseline_app_bank5_freezecrowd_wapp125_crowd055_nms90_det012_relu_cascade_minhits2_adaptcentral_ar24 \
  --reid_ckpt outputs/reid_mamba_temporal/reid_best.pt \
  --axis_mode none \
  --output_mode all
```

## Baseline tracker evaluation

```bash
python scripts/run_tracker_baselines_trackeval.py \
  --trackers bytetrack ocsort botsort deepocsort \
  --reid_ckpt outputs/reid_mamba_temporal/reid_best.pt \
  --axis_mode none
```

## Speed benchmark

```bash
python scripts/benchmark_tracker_speed.py \
  --reid_ckpt outputs/reid_mamba_temporal/reid_best.pt \
  --max_frames 200
```

## Outputs

- Temporal ReID checkpoints: `outputs/reid_mamba_temporal/`
- Ablation logs and CSVs: `outputs/ablation_temporal/`
- Tracker metrics CSVs: `outputs/*trackeval*.csv`
- Speed benchmark CSV: `outputs/benchmark_speed/tracker_speed.csv`

