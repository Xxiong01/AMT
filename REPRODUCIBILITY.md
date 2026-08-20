# Reproduction Guide

This guide maps the manuscript tables to the code, configuration, and commands
that generate them. Run all commands from the repository root.

## 1. Frozen software and evaluator

Reference environment:

- Ubuntu 22.04 / WSL2;
- Python 3.10.18;
- PyTorch 2.5.1+cu121 and torchvision 0.20.1+cu121;
- CUDA runtime 12.1;
- NumPy 1.26.4, SciPy 1.15.3, Pillow 11.0.0, PyYAML 6.0.2;
- mamba-ssm 2.2.5 and causal-conv1d 1.5.2;
- psutil 7.0.0 for runtime monitoring;
- official TrackEval commit
  `12c8791b303e0a0b50f753af204249e622d0281a`.

Create the environment and evaluator exactly as shown in `README.md`, then run:

```bash
python scripts/validate_release.py
```

## 2. Data and checkpoint-selection protocol

Edit only the path fields in `configs/datasets/mft25.yaml`. Do not change the
declared sequence list, GT filenames, or detection filename when reproducing
the paper.

```bash
python scripts/prepare_reid_dev_split.py \
  --dataset-config configs/datasets/mft25.yaml \
  --output-root data/MFT25-reid-protocol
```

The script reserves the last 20% of frame indices from each MFT25 Train
sequence as a deterministic development subset. Val is not read during ReID
training or checkpoint selection. `best.pt` is selected by development
cross-entropy only.

Train a single model/seed with:

```bash
python scripts/train_temporal_reid.py \
  --model-config configs/models/mamba_l48.yaml \
  --seed 0 --device cuda \
  --output-dir outputs/training/mamba_l48/seed_0
```

Train seeds 0, 1, and 2 for any declared model with:

```bash
python scripts/train_seed_batch.py \
  --model-config configs/models/mamba_l48.yaml \
  --output-dir outputs/training/mamba_l48 \
  --seeds 0 1 2 --device cuda
```

## 3. Tables 2 and 6: main and per-sequence results

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

Use `official_trackeval_metrics.csv` for the AMT row of Table 2 and
`official_trackeval_per_sequence.csv` for Table 6.

The three external baselines are run with their official implementations under
the same MFT25 detections and score threshold. To apply the identical evaluator
to an external method's eight MOT files:

```bash
python scripts/evaluate_trackeval.py \
  --dataset-config configs/datasets/mft25.yaml \
  --mot-results-dir /path/to/ByteTrack_mot_files \
  --tracker-name ByteTrack \
  --output-dir outputs/baselines/ByteTrack \
  --trackeval-root external_tools/TrackEval
```

Replace the directory and tracker name for OC-SORT and SU-T. Each input
directory must contain `<sequence>.txt` for all eight declared sequences.

## 4. Table 3: cumulative construction

```bash
python scripts/run_config_batch.py \
  --config-glob 'configs/experiments/cumulative/C*.yaml' \
  --checkpoint checkpoints/best.pt \
  --output-dir outputs/table3_cumulative \
  --trackeval-root external_tools/TrackEval --device cuda

python scripts/summarize_experiments.py \
  --runs-root outputs/table3_cumulative \
  --output outputs/table3_cumulative/summary.csv
```

## 5. Table 4: leave-one-out ablation

```bash
python scripts/run_config_batch.py \
  --config-glob 'configs/experiments/leave_one_out/*.yaml' \
  --checkpoint checkpoints/best.pt \
  --output-dir outputs/table4_leave_one_out \
  --trackeval-root external_tools/TrackEval --device cuda

python scripts/summarize_experiments.py \
  --runs-root outputs/table4_leave_one_out \
  --output outputs/table4_leave_one_out/summary.csv
```

## 6. Table 5: write policy and temporal-input construction

```bash
python scripts/run_config_batch.py \
  --config-glob 'configs/experiments/write_policy/W*.yaml' \
  --checkpoint checkpoints/best.pt \
  --output-dir outputs/table5_write_policy \
  --trackeval-root external_tools/TrackEval --device cuda

python scripts/run_config_batch.py \
  --config-glob 'configs/experiments/history_construction/*.yaml' \
  --checkpoint checkpoints/best.pt \
  --output-dir outputs/table5_history_construction \
  --trackeval-root external_tools/TrackEval --device cuda
```

Generate the GT-only write-contamination and effective-depth diagnostics after
tracking. Ground truth is used only by this offline diagnostic:

```bash
python scripts/diagnose_history_reliability.py \
  --run-dir outputs/table5_write_policy/W3/seed_0 \
  --dataset-config configs/datasets/mft25.yaml \
  --output-dir outputs/table5_write_policy/W3_diagnostics
```

Repeat with the W0 run for the unfiltered-write comparison.

## 7. Table 7: paired sequence-level statistics

Prepare one CSV with columns `method,sequence,HOTA,AssA,IDF1,IDSW`, containing
AMT, `SU-T w/ ReID`, and OC-SORT rows for the eight sequences. Then run:

```bash
python scripts/compute_paired_statistics.py \
  --config configs/experiments/paired_statistics/wilcoxon.yaml \
  --input /path/to/per_sequence_methods.csv \
  --reference-method AMT \
  --output outputs/table7_paired_statistics.csv
```

The script defines IDSW improvement as baseline minus AMT and all other gains
as AMT minus baseline. Tests are two-sided, exploratory, and unadjusted.

## 8. Table 8: three-seed encoder comparison

Train each encoder independently:

```bash
python scripts/train_seed_batch.py --model-config configs/models/mean_l48.yaml --output-dir outputs/training/mean_l48 --seeds 0 1 2 --device cuda
python scripts/train_seed_batch.py --model-config configs/models/gru_l48.yaml --output-dir outputs/training/gru_l48 --seeds 0 1 2 --device cuda
python scripts/train_seed_batch.py --model-config configs/models/lstm_l48.yaml --output-dir outputs/training/lstm_l48 --seeds 0 1 2 --device cuda
python scripts/train_seed_batch.py --model-config configs/models/transformer_l48.yaml --output-dir outputs/training/transformer_l48 --seeds 0 1 2 --device cuda
python scripts/train_seed_batch.py --model-config configs/models/mamba_l48.yaml --output-dir outputs/training/mamba_l48 --seeds 0 1 2 --device cuda
```

Evaluate and summarize:

```bash
python scripts/run_config_batch.py \
  --config-glob 'configs/experiments/encoders/*.yaml' \
  --output-dir outputs/table8_encoders \
  --trackeval-root external_tools/TrackEval --device cuda

python scripts/summarize_experiments.py \
  --runs-root outputs/table8_encoders \
  --output outputs/table8_encoders/mean_sample_sd.csv
```

The optional single-frame control is declared in the same folder but is not a
row of the five-encoder manuscript table.

## 9. Table 9: three-seed temporal length

Train `mamba_l8.yaml`, `mamba_l16.yaml`, `mamba_l32.yaml`, `mamba_l48.yaml`, and
`mamba_l64.yaml` with `train_seed_batch.py`, writing to the corresponding
`outputs/training/mamba_l*/` directory. Then run:

```bash
python scripts/run_config_batch.py \
  --config-glob 'configs/experiments/lengths/L*.yaml' \
  --output-dir outputs/table9_lengths \
  --trackeval-root external_tools/TrackEval --device cuda

python scripts/summarize_experiments.py \
  --runs-root outputs/table9_lengths \
  --output outputs/table9_lengths/mean_sample_sd.csv
```

## 10. Table 10: HOTA/IDF1-IDSW operating points

```bash
python scripts/run_config_batch.py \
  --config-glob 'configs/experiments/operating_points/P*.yaml' \
  --checkpoint checkpoints/best.pt \
  --output-dir outputs/table10_operating_points \
  --trackeval-root external_tools/TrackEval --device cuda
```

`P5_geometry_margin.yaml` is the one-factor geometry-margin alternative; the
other files are the five joint threshold presets.

## 11. Table 11: internal cross-scene case study

Instantiate `configs/datasets/cct_template.yaml` with the private sequence IDs
and paths, then run `scripts/run_experiment.py` with
`configs/experiments/external_validation/cct_zero_shot.yaml`. The internal
clips, annotations, detections, and baseline trajectories are available from
the corresponding author upon reasonable request.

## 12. Table 12 and Supplementary Table S3: OFAT sensitivity

```bash
python scripts/run_ofat_sensitivity.py \
  --experiment-config configs/experiments/hyperparameter_sensitivity/ofat.yaml \
  --checkpoint checkpoints/best.pt \
  --output-dir outputs/table12_ofat \
  --trackeval-root external_tools/TrackEval --device cuda

python scripts/summarize_experiments.py \
  --runs-root outputs/table12_ofat \
  --output outputs/table12_ofat/summary.csv
```

The YAML contains the exact seven three-value grids used in the revised paper,
including the geometry-confidence margin.

## 13. Table 13: cached and cold-cache runtime

Build a checkpoint-specific current-frame cache:

```bash
python scripts/build_embedding_cache.py \
  --dataset-config configs/datasets/mft25.yaml \
  --tracker-config configs/tracker/amt_l48.yaml \
  --checkpoint checkpoints/best.pt \
  --minimum-score 0.0 --device cuda \
  --output-dir outputs/cache/amt_seed0
```

Run three cached repetitions for F00, F10, F11, and R1. Example for F11:

```bash
python scripts/benchmark_runtime.py \
  --experiment-config configs/experiments/factorial/F11.yaml \
  --checkpoint checkpoints/best.pt \
  --embedding-cache-dir outputs/cache/amt_seed0 \
  --mode cached --repetitions 3 \
  --output-dir outputs/runtime/cached_F11 \
  --trackeval-root external_tools/TrackEval --device cuda
```

For the other cached rows, substitute F00, F10, and
`history_construction/R1.yaml`. Run cold repetitions for F00, F11, and R1 by
removing `--embedding-cache-dir`, changing `--mode cold`, and using a new output
directory. `trajectory_generation_seconds` excludes TrackEval; the separate
process overhead field records evaluation and process-launch time.

## 14. Supplementary robustness experiments

```bash
python scripts/run_config_batch.py --config-glob 'configs/experiments/thresholds/*.yaml' --checkpoint checkpoints/best.pt --output-dir outputs/thresholds --trackeval-root external_tools/TrackEval --device cuda
python scripts/run_config_batch.py --config-glob 'configs/experiments/detection_dropout/*.yaml' --checkpoint checkpoints/best.pt --output-dir outputs/dropout --trackeval-root external_tools/TrackEval --device cuda
```

Detection-dropout configs declare perturbation seeds 0, 1, and 2. The batch
runner uses those seeds automatically.

## 15. Output and provenance checks

Every tracking run writes its resolved configuration, checkpoint metadata,
official aggregate and per-sequence metrics, MOT files, diagnostic events, and
timing summaries beneath the selected output directory. Embedding caches are
accepted only when checkpoint, dataset-config, and tracker-config hashes match
the active run. Do not mix caches across checkpoints or experiment families.
