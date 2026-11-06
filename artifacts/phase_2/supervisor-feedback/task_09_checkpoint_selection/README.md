# Task 09: checkpoint selection by val loss, warmup excluded (06-11-2026)

Supervisor request 1 (06-11-2026): "Sửa code chọn checkpoint: bỏ qua 5 epoch
warmup, chọn theo val loss, patience 15. Chạy thử 1 run để kiểm tra code."
It answers open question 4 / section 6.2 of
`notebooks/phase 2/supervisor-feedback/SUPERVISOR_REPORT_2026-11-01.md`
(ConvNeXt fold 3 picked epoch 3, Swin fold 5 picked epoch 5).

## Code

Commit `22db43a` (branch `feat/supervisor-2026-11-06`),
`notebooks/phase 2/supervisor-feedback/task_01_fivefold_cv_training/train_kfold.py`:

```
--selection_metric val_loss --select_after_warmup --early_stopping_patience 15
```

- Epochs 1..warmup_epochs (5) are trained and logged, but they can neither
  become `best.pth` nor count towards patience. `history.json` marks them
  `"eligible": false`.
- `best.pth` = lowest inner-val loss. The loss is the training criterion, i.e.
  cross-entropy with label smoothing 0.1.
- The LR schedule is unchanged (5 epochs linear warmup + cosine). Removing the
  warmup itself is a separate request (task 8).
- The default is still the Final-B rule (best val accuracy from epoch 1,
  patience 10). Config keys for the new rule are only added when it is used,
  so the 01-11-2026 CV v2 fingerprints are unchanged.
- Not yet changed: `src/train/trainer.py` (`TrainingEngine.fit`, used by
  Final-B and LR selection). GitNexus impact risk is HIGH, and it is planned
  for the Final-B rerun (task 6).

## Trial run `trial-20261106T042955Z/`

ConvNeXt-Tiny, CV v2 fold 3, backbone LR 1e-4, 17 species, RTX 3060 (local),
conda env `wood` (`pip_freeze.txt`, `conda_env.yaml`). Data:
`data/task-B-group-split-224`, rebuilt locally with
`scripts/build_resized_cache.py`. All 4,800 files match
`docs/timeline/01-11-2026-task-B-group-split-224.sha256`, so the input bytes
are identical to the Vast run.

```zsh
cd "notebooks/phase 2/supervisor-feedback/task_01_fivefold_cv_training"
python train_kfold.py --backbone convnext_tiny --backbone_lr 1e-4 --fold 3 \
  --data_root ../../../../data/task-B-group-split-224 \
  --output_root ../../../../data/experiments/Task-B-origin-cv-selection-test \
  --selection_metric val_loss --select_after_warmup --early_stopping_patience 15
```

The run recorded `git_commit` 45f71ca because the code was not committed yet.
Its `config.code_sha256` (`7f83ef03…`) recomputes to the same value from the
files in `22db43a`, so the code that ran is the committed code.

Checks (recomputed from `history.json`, `metrics.json` and `best.pth`):

| Check | Result |
|---|---|
| Warmup epochs 1–5 never marked best | yes. Epoch 3 (val acc 99.24%) was skipped |
| best epoch = argmin val loss over epochs 6+ = `best.pth` epoch | 19 = 19 = 19 |
| Early stop = best + patience | stopped at 34 = 19 + 15 |
| Same train/val/test origins as 01-11 fold 3 | yes (2,892 / 927 / 981 images) |

Comparison with 01-11 fold 3 (`../task_01_fivefold_cv_training/results-v2/convnext_tiny/fold_3/`):

| Rule | Best epoch | Val acc | Val loss | Epochs | Held-out acc | Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| 01-11: val acc, epoch 1+, patience 10 (RTX 4090) | 3 | 99.24 | 0.6451 | 13 | 98.47 | 98.42 |
| 06-11: val loss, epoch 6+, patience 15 (RTX 3060) | 19 | 99.46 | 0.5900 | 34 | 98.88 | 98.72 |

The held-out difference is 4/981 images on one fold, and the two runs ran on
different GPUs. Epoch 1 is identical to 4 decimals, but the runs diverge from
epoch 2 (cuDNN/AMP non-determinism). This is a code check, not evidence that
one rule gives better accuracy.

## Checkpoint

`best.pth` (111 MB) stays local at
`data/experiments/Task-B-origin-cv-selection-test/convnext_tiny/fold_3/best.pth`
(git-ignored). Its SHA-256 is in `trial-20261106T042955Z/checkpoints.sha256`.
It has not been uploaded to Drive.
