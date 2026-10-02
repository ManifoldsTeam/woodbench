# Phase 2: Supervisor Feedback Tasks

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
