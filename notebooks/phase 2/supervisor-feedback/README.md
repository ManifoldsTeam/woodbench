# Phase 2: Supervisor Feedback Tasks

## Output location (updated 2026-10-13)

Notebook/script sources stay here. Generated CSV, figures, inference runs,
registries, summary exports and launcher logs now live in
[`artifacts/phase_2/supervisor-feedback/`](../../../artifacts/phase_2/supervisor-feedback/).
The original output paths are relative compatibility symlinks so immutable
historical registries and existing consumers still resolve. Do not replace
these symlinks with new copies. New notebook exports use the canonical path.
Training checkpoints and CV/LR-tuning outputs already under `data/experiments/`
are unchanged. See the artifact README and relocation manifest for file hashes.
This local reorganization is not authorization to publish images or metadata.

## Follow-up requests (01-11-2026) — done

Report: [SUPERVISOR_REPORT_2026-11-01.md](SUPERVISOR_REPORT_2026-11-01.md). Context:
[handoff 01-11-2026](../../../docs/timeline/01-11-2026-handoff-vast.md).

| Step | Request | Code | Result |
|---|---|---|---|
| 1 | Backbone LR 1e-5 vs 1e-4 on val | [task_07_lr_selection](task_07_lr_selection/) | 1e-4 for all 5 models |
| 2 | Origin CV v2, 5 models × 5 folds, best-val, per-species F1 | [task_01](task_01_fivefold_cv_training/) | ConvNeXt 99,13 ± 0,39 … MobileNet 96,61 ± 0,88 |
| 3 | ConvNeXt retrained on grayscale, fold 1 | task_01 `--input_variant grayscale` | 94,68% (RGB 99,17%) |
| 4 | ConvNeXt retrained on 16px, fold 1 | task_01 `--input_variant px16` | 92,81% |
| 5 | Latency/VRAM of new checkpoints | [task_08_efficiency_rtx4090](task_08_efficiency_rtx4090/) | RTX 4090; unchanged vs random weights |

All runs read `data/task-B-group-split-224`, a lossless pre-resized copy made by
`scripts/build_resized_cache.py` (identical model input, ~20× faster on CPU-limited hosts).

## 6 Tasks from Supervisor (30-09-2026)

| # | Task | Type | Status | Time |
|---|------|------|--------|------|
| 1 | 5-fold CV by origin key | Training | 📝 Ready | 10+ hours |
| 2 | Robustness tests | Inference | ✅ Done | 1-2h |
| 3 | Dark artifact check | Inspection | 📝 Ready | 1-2h |
| 4 | Degradation re-run | Inference | ✅ Done | 1-2h |
| 5 | ResNet LR 1e-4 | Training | 📝 Ready | 5-10h |
| 6 | Error analysis | Analysis | ✅ Done | 1-2h |

## Completed Tasks ✅

### Task 2: Robustness Tests
- **Results:** Model relies heavily on COLOR (-26% with grayscale)
- Model does NOT rely on edges (-0.4% with edge mask)
- Model is sensitive to spatial layout

### Task 4: Degradation Re-run
- **Results:** 16×16 resize causes -71% accuracy drop
- Blur causes -42% accuracy drop
- Model needs fine-grained spatial details

### Task 6: Error Analysis
- **Results:** ALL 125 P.krempfii→P.dalatensis errors from ONE origin
- Specimen-specific issue, NOT model weakness

## Ready to Run 📝

### Task 3: Dark Artifact Check (1-2h)
```bash
conda activate wood
jupyter notebook "task_03_dark_artifact_check/dark_artifact_check.ipynb"
```

### Task 5: ResNet LR Tuning (5-10h)
```bash
conda activate wood
jupyter notebook "task_05_resnet_lr_tuning/resnet_lr_tuning.ipynb"
```

### Task 1: 5-Fold CV (10+ hours) - TRAINING
- ConvNeXt + MobileNet × 5 folds = 10 training runs minimum
- Will produce proper confidence intervals
```bash
conda activate wood
jupyter notebook "task_01_fivefold_cv_training/fivefold_cv_training.ipynb"
```

## System Requirements Check ✅

- GPU: RTX 3060 (12.5 GB VRAM)
- Dataset: 3365 train, 903 val, 532 test images
- Classes: 17 species
- All models available in timm
