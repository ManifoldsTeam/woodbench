# Task 15 outputs: 14-species runs (supervisor requests 6-7)

Run on Vast (RTX 4090) on 07-11-2026, 07:06–08:40 UTC, branch
`feat/supervisor-2026-11-06` at `a496be4`. Code and run rules:
`notebooks/phase 2/supervisor-feedback/task_15_clean14_runs/README.md`.
Runbook and results: `docs/timeline/07-11-2026-handoff-vast-clean14.md` (section 9).

| Path | Content |
|---|---|
| `clean14_runs_executed.ipynb` | Executed notebook: environment, launcher output, a progress snapshot every 5 minutes (running epochs, each finished run with its result and files), tables |
| `results/summary.md` | All tables |
| `results/cv_summary.csv`, `cv_folds.csv` | CV, 5 models × 5 folds, held-out fold scored once with `best.pth` |
| `results/cv_vs_17species.csv` | Same folds and 14 species: CV v2 17-species model vs this CV |
| `results/degraded_summary.csv` | ConvNeXt rgb / grayscale / px16 per fold |
| `results/final_b_runs.csv`, `final_b_summary.csv`, `final_b_per_species_f1.csv` | Final-B, 5 models × 3 seeds, test 432 images |
| `results/checkpoints.sha256` | sha256 of the 50 `best.pth` (repo-relative paths) |
| `results/runs/` | json/csv of every run (no `.pth`) |
| `logs/` | Launcher output, per-job logs and exit codes, `pip_freeze.txt`, `nvidia-smi.txt`, papermill log |

Headline (mean ± SD, %): CV ConvNeXt 99.44 ± 0.31, Swin 99.34 ± 0.26,
EfficientNet 97.57 ± 0.93, ResNet 97.12 ± 1.35, MobileNet 95.29 ± 1.34;
Final-B test Swin 99.69 ± 0.13, ConvNeXt 98.92 ± 0.27, MobileNet 98.53 ± 0.27,
EfficientNet 97.92 ± 0.46, ResNet 97.53 ± 0.81; ConvNeXt CV grayscale
94.83 ± 2.14, 16px 90.90 ± 0.90. EfficientNet, MobileNet and some ResNet runs
reach the 50-epoch cap with the best epoch ≥ 45.

Checkpoints are not in git. Backup: `drive:wood/2026-11-07-clean14-runs/`
(one zip per experiment folder, `zip.md5`, `checkpoints.sha256`, `README.txt`).
Unzip into `woodbench/data/experiments/`.
