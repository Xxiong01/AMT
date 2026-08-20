# AquaMambaTrack (AMT)

This repository contains the paper-aligned implementation of AquaMambaTrack,
including temporal ReID training, online reliability-gated per-track FIFO
inference, controlled ablations, official TrackEval evaluation, diagnostics,
and runtime measurement.

The central implementation rule is explicit: a successful association updates
the track's geometry and motion state, but it does not automatically authorize
the current frame feature to enter temporal identity memory. Only eligible
matches that pass the declared geometry, appearance, and crowd checks are
written to the per-track FIFO. Current detections are always encoded from their
current crop as a one-element temporal query; the main AMT path does not build
IoU-linked histories before tracking and has no conditional single-frame
fallback.

## Reference environment

The numerical results were produced in the Linux environment frozen in
`environment.yml`: Python 3.10.18, PyTorch 2.5.1 with CUDA 12.1,
torchvision 0.20.1, mamba-ssm 2.2.5, and causal-conv1d 1.5.2. Linux or WSL2 is
recommended because the Mamba CUDA extensions are not a native Windows target.

```bash
conda env create -f environment.yml
conda activate amt-paper
pip install -e . --no-deps
```

If the two CUDA extensions must be rebuilt, install PyTorch first and then run:

```bash
pip install causal-conv1d==1.5.2 --no-build-isolation
pip install mamba-ssm==2.2.5 --no-build-isolation
```

## Official TrackEval

All paper metrics use the official TrackEval repository at the pinned revision
below. The evaluator checks the Git revision and stops if it differs.

```bash
git clone https://github.com/JonathonLuiten/TrackEval external_tools/TrackEval
git -C external_tools/TrackEval checkout 12c8791b303e0a0b50f753af204249e622d0281a
```

## Data layout

MFT25 is not redistributed. Update `data_root` in
`configs/datasets/mft25.yaml` if necessary. The expected files are:

```text
data/MFT25-train/<sequence>/
  img1/000001.jpg ...
  det/det_yolox_ckpt.txt
  gt/gt.txt
  gt/gt_train_half.txt
  gt/gt_val_half.txt
```

The final Val split is not used for checkpoint selection. Create the fixed
Train-derived development protocol with:

```bash
python scripts/prepare_reid_dev_split.py \
  --dataset-config configs/datasets/mft25.yaml \
  --output-root data/MFT25-reid-protocol
```

## Validate the release

```bash
python scripts/validate_release.py
```

This checks Python and YAML syntax, the frozen main configuration, checkpoint
metadata and T=1/T=48 forward passes, online-memory invariants, controlled
experiment semantics, the paper OFAT grid, and the official TrackEval policy.
Before the CUDA extensions are installed, the non-forward checks can be run
with `python scripts/validate_release.py --static-only`.

## Reproduce the main AMT result

```bash
python scripts/track_amt.py \
  --dataset-config configs/datasets/mft25.yaml \
  --tracker-config configs/tracker/amt_l48.yaml \
  --checkpoint checkpoints/best.pt \
  --output-dir outputs/main/seed_0 \
  --seed 0 --device cuda --batch-size 128

python scripts/evaluate_trackeval.py \
  --dataset-config configs/datasets/mft25.yaml \
  --output-dir outputs/main/seed_0 \
  --trackeval-root external_tools/TrackEval \
  --tracker-name AMT
```

The frozen result is 53.473 HOTA, 69.982 DetA, 41.112 AssA, 59.543 IDF1,
85.265 MOTA, and 507 IDSW. Per-sequence results are written by the same command.

Complete table-by-table commands, output locations, training commands, baseline
MOT-file evaluation, diagnostics, and timing instructions are in
`REPRODUCIBILITY.md`.

## Repository map

- `fishmambatrack/runtime/online_amt.py`: main current-frame and online FIFO path.
- `fishmambatrack/tracking/amt_tracker.py`: cascade, reliability decisions, and
  re-activation.
- `experiments/`: controlled variants used only for paper comparisons.
- `configs/models/`: independently trainable encoder and temporal-length models.
- `configs/experiments/`: cumulative, leave-one-out, history, write-policy,
  robustness, sensitivity, operating-point, and efficiency declarations.
- `scripts/`: split preparation, training, tracking, TrackEval, batching,
  diagnostics, statistics, summarization, cache building, and timing.

ByteTrack, OC-SORT, and SU-T remain external methods and are not vendored. Their
MOT text outputs can be evaluated under the same pinned TrackEval protocol with
the generic staging option documented in `REPRODUCIBILITY.md`.
