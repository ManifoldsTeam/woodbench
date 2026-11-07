# Task 14: does warmup explain why the August result does not reproduce? (07-11-2026)

Supervisor request 8 (06-11-2026): "chạy lại ConvNeXt seed 42 trên Final-B cũ, LR 1e-5, không có warmup
(1–2 run). Mục đích là xác nhận warmup có phải nguyên nhân khiến kết quả cũ không tái lập được."

Notebook: `notebooks/phase 2/supervisor-feedback/task_14_august_reproduction/august_reproduction.ipynb`
(source, outputs cleared). Executed copy: `august_reproduction_executed.ipynb`. RTX 3060, conda env `wood`
(torch 2.15.0+cu132, timm 1.0.30), `data/task-B-group-split-224` (all 4,800 files match the 01-11 sha256 list).
Split: Final-B 17 species, 3,365 / 903 / 532.

## Answer

**No, warmup is not the cause.** The August code, which has warmup 5, gives the August test result again on
another GPU: 99.81%, the same single error. What differs between August and the 01-11 rerun is the effective
learning rate. The August code uses an old name-based parameter grouping that puts 25.9 M of the 27.8 M
ConvNeXt weights (93%) at 1e-4, so "backbone LR 1e-5" was only nominal. The 01-11 rerun used a true 1e-5.

| Run | Effective LR | Warmup | GPU | Best epoch | Val acc | Test acc | Test macro-F1 | Test errors |
|---|---|---:|---|---:|---:|---:|---:|---:|
| August original (`Task-B-new/19-08-2026`) | 93% of weights at 1e-4 | 5 | 4090 | 14 | 99.89 | **99.81** | 99.83 | 1 |
| **Run 2**: August code `75df6c0`, unchanged | 93% of weights at 1e-4 | 5 | 3060 | 10 | 99.67 | **99.81** | 99.83 | 1 |
| 01-11 LR selection `bblr_1e-5` | true 1e-5 | 5 | 4090 | 10 | 99.89 | 98.68 | 98.79 | 7 |
| **Run 1**: as requested | true 1e-5 | **0** | 3060 | 21 | 99.89 | 99.25 | 99.31 | 4 |

All rows are scored with the same function (`train_lr_select.score`, FP32) from each `best.pth`. The 01-11
checkpoint is on Drive only, so that row is taken from its `metrics.json`, which was scored with the same function.

- **Run 2 vs August:** same test accuracy, F1 and misclassified image (`3384. Pinus krempfii.4 (130)` ->
  *P. dalatensis*). The best epoch and val accuracy differ (non-determinism, other GPU). The environment is not
  the reason the August number could not be reproduced, so no Vast run is needed.
- **Run 1 vs 01-11:** with a true 1e-5, removing warmup gives 4 errors instead of 7. That is 3 test images in one run per
  arm on different GPUs, so it is not evidence that warmup hurts; it stays below the August code either way.
- **Where the errors are:** 10 of the 12 test errors above are photos 129–134 of specimen 3384.4 (*P. krempfii*)
  predicted as *P. dalatensis*. This is the same block found in task 11. The other two are
  `3386. Podocarpus pilgeri.1 (29)` -> *Chamaecyparis obtusa*.
- **Validation is nearly saturated:** the top epochs differ by one of 903 val images (`val_saturation.csv`,
  `val_curves.png`), so which epoch becomes `best.pth` is close to arbitrary. Run 1 reached 100% val under AMP
  during training, which is 99.89% when re-scored in FP32.

Consequence for the paper: the August Final-B table was produced with an effective LR of about 1e-4, not
"backbone 1e-5, head 1e-4". This agrees with the 01-11 LR selection, which chose 1e-4 for every model. The
config description in the paper should say so (see `SUPERVISOR_REPORT_2026-11-01.md`, section 1).

## How run 2 was run

A detached git worktree of commit `75df6c0` (the August training code) was created under
`data/experiments/Task-B-august-repro/` and removed afterwards. A child process called its
`final_b_17class.train_one(experiment="main", seed=42, backbone="convnext_tiny")`. The only addition is a
wrapper around `TrainingEngine.fit` that writes `history.json` and `param_groups.json` after `fit` returns.
`run2/test_metrics_final_epoch.json` is the August script's own evaluation of the final-epoch model (99.25%), not
of `best.pth`. The registry numbers always came from `best.pth`.

## Files

| File | Content |
|---|---|
| `results.csv` | Table above, with val/test loss and checkpoint sha256 |
| `val_saturation.csv`, `val_curves.png` | Per-run val accuracy at and near the maximum; val accuracy/loss curves |
| `predictions_test_*.csv` | Test predictions of the August, run 2 and run 1 checkpoints |
| `run1/`, `run2/` | run config, per-epoch history, metrics, training log; run 2 also `param_groups.json` |
| `environment.json`, `checkpoints.sha256` | Software/GPU record; checkpoint hashes |

The checkpoints stay local in `data/experiments/Task-B-august-repro/` (git-ignored, 1.3 GB including `last.pth` and optimizer state) and
are not on Drive.
