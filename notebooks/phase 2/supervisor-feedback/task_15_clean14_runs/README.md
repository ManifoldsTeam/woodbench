# Task 15: 14-species runs (supervisor requests 6-7, 06-11-2026)

Runbook for Vast: [`docs/timeline/07-11-2026-handoff-vast-clean14.md`](../../../../docs/timeline/07-11-2026-handoff-vast-clean14.md).

Data: the 14 species without Callitris columellaris, Cunninghamia lanceolata
and Fokienia hodginsii (4,115 images, 140 origins). The images are read from the
224x224 cache `data/task-B-group-split-224`, which holds all 17 species. The
code drops the 3 species.

| Request | Runs | Script | Output (`data/experiments/`) |
|---|---:|---|---|
| 6, CV | 5 models × 5 folds | `../task_01_fivefold_cv_training/train_kfold.py --species_set clean14` | `Task-B-clean14-cv/` |
| 6, Final-B | 5 models × seeds 42/123/456 | `train_final_b.py` | `Task-B-clean14-final/seed_<s>/<model>/` |
| 7 | ConvNeXt × 5 folds × {grayscale, px16} | `train_kfold.py --input_variant ...` | `Task-B-clean14-cv-{grayscale,px16}/` |

`run_clean14.sh` queues all 22 jobs (50 runs) with at most `MAX_PARALLEL` jobs
at a time, and `summarize_clean14.py` writes the tables to
`artifacts/phase_2/supervisor-feedback/task_15_clean14_runs/results/`.

## The same rule for every run

- Backbone and head LR are both 1e-4; the head is `model.get_classifier()`.
- 5 epochs of linear warmup, then cosine; at most 50 epochs.
- `best.pth` is the epoch with the lowest val loss among epochs 6 and later.
  Training stops early after 15 epochs without improvement (patience 15).
- The held-out fold, or val and test for Final-B, is scored once, from `best.pth`.
- Everything else follows Final-B main: AdamW with weight decay 1e-4, label
  smoothing 0.1, AMP, a batch size per backbone, and the transforms in
  `src/data/transforms.py`.

## Folds

`make_origin_folds` shuffles the origins of each species in turn, all from one
RNG stream. Drawing folds on the 14 species directly would therefore give every
species after the first removed one different folds. Instead, `--species_set
clean14` draws the 17-species CV v2 folds, then removes the 3 species
(`origin_cv.drop_species`).

Checked on the real data: for each of the 5 folds, the train, inner-val and
held-out origins of the 14 species match the `metrics.json` of CV v2 from
01-11. The folds are 2466–2500 train, 800–843 val and 813–847 held-out images.
Because the folds match, `summarize_clean14.py` can compare the two runs fold by
fold (`cv_vs_17species.csv`).

## Final-B

`train_final_b.py` uses the training loop of `train_kfold.py`
(`train_epoch`/`validate`) and the scoring of `train_lr_select.py` (`score`, FP32).
`src/train/trainer.py` is unchanged. Before training, the script checks every
image against `docs/timeline/01-11-2026-task-B-group-split-224.sha256`. The
split is train 2,880 / val 803 / test 432 images.

How reruns behave:

- If `metrics.json` exists, the run is skipped.
- If the run folder exists without `metrics.json`, it is deleted and the run
  starts again from epoch 1.

The val numbers in the training log are AMP batch means. The val numbers in
`metrics.json` are FP32 per-image means, so they can differ slightly; for
example, in the smoke test val acc was 89.04% in the log and 88.92% in
`metrics.json`.

## Local checks (07-11-2026, RTX 3060)

- `pytest tests/test_clean14_runs.py tests/test_origin_cv.py tests/test_cv_training_runs.py`:
  69 passed.
- Dry-runs pass for all 5 backbones with both scripts.
- Smoke runs (not results):
  - Final-B MobileNet, 4 epochs: epoch 1 was skipped for `best.pth`. Two runs
    gave identical epochs. A completed run is skipped and an unfinished one
    restarts.
  - CV clean14, 2 epochs × 5 folds: MobileNet RGB, ConvNeXt RGB and ConvNeXt
    grayscale.
  - `summarize_clean14.py` ran on these smoke outputs.
- Launcher, tested with a fake Python:
  - It starts 22 jobs and never runs more than `MAX_PARALLEL` at once.
  - It exits 1 and reports how many jobs failed.

## Vast run (07-11-2026, RTX 4090)

`clean14_runs.ipynb` is the record of the run. It starts `run_clean14.sh`
(never a second one while one is running), prints launcher events, the latest
epoch of each running job and every finished run with its result and files
every 5 minutes, then runs `summarize_clean14.py` and copies the logs. Run it
headless with papermill (command in its first cell); the executed copy is
`artifacts/phase_2/supervisor-feedback/task_15_clean14_runs/clean14_runs_executed.ipynb`.

- 07:06–08:40 UTC, 22/22 jobs exit 0, 50/50 runs; torch 2.14.1+cu126 (the host
  driver supports CUDA 12.7 only, so the cu132 build of 2.15.0 cannot use the GPU).
- `pytest`: 68 passed, 1 skipped (needs the full-resolution images).
- Results, deviations and caveats: handoff section 9 and
  `docs/timeline/07-11-2026.md`. Checkpoints: `drive:wood/2026-11-07-clean14-runs/`.
