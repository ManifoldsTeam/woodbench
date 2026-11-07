# 14-species runs (supervisor requests 6-7, 06-11-2026)

CV report `Task-B-clean14-cv`: `cv-20261107T084125Z-5a586a35`

CV report `Task-B-clean14-cv-grayscale`: `cv-20261107T084125Z-0fa52558`

CV report `Task-B-clean14-cv-px16`: `cv-20261107T084126Z-9f17c7a8`

## CV, 5 models × 5 folds (held-out fold, best.pth)

| Model | Acc (%) | Macro-F1 (%) | OOF acc (%) | Best epoch | Epochs |
|---|---:|---:|---:|---:|---:|
| convnext_tiny | 99.44 ± 0.31 | 99.46 ± 0.28 | 99.44 | 17.4 | 32.4 |
| swin_tiny | 99.34 ± 0.26 | 99.33 ± 0.24 | 99.34 | 33.4 | 47.2 |
| efficientnet_b3 | 97.57 ± 0.93 | 97.51 ± 0.92 | 97.57 | 48.0 | 50.0 |
| resnet50 | 97.12 ± 1.35 | 97.02 ± 1.33 | 97.13 | 39.2 | 49.6 |
| mobilenetv3_large | 95.29 ± 1.34 | 94.91 ± 1.45 | 95.29 | 46.4 | 50.0 |

## Same folds, same 14 species: 17-species model (CV v2, 01-11 rule) vs 14-species model

CV v2 accuracy counts a prediction of one of the 3 removed species as wrong.

| Model | CV v2 on 14 species (%) | 14-species CV (%) | Mean difference (pts) |
|---|---:|---:|---:|
| convnext_tiny | 99.03 ± 0.41 | 99.44 ± 0.31 | +0.41 |
| swin_tiny | 98.88 ± 1.21 | 99.34 ± 0.26 | +0.47 |
| efficientnet_b3 | 97.47 ± 1.14 | 97.57 ± 0.93 | +0.09 |
| resnet50 | 96.89 ± 1.27 | 97.12 ± 1.35 | +0.23 |
| mobilenetv3_large | 96.24 ± 1.17 | 95.29 ± 1.34 | -0.95 |

## ConvNeXt-Tiny, input degradation (same 5 folds)

| Input | Acc per fold (%) | Acc (%) | Macro-F1 (%) | Best epoch |
|---|---|---:|---:|---:|
| rgb | 99.26, 99.15, 99.65, 99.88, 99.26 | 99.44 ± 0.31 | 99.46 ± 0.28 | 17.4 |
| grayscale | 91.79, 93.69, 95.16, 96.31, 97.18 | 94.83 ± 2.14 | 94.50 ± 2.23 | 27.8 |
| px16 | 90.32, 91.38, 89.61, 91.39, 91.78 | 90.90 ± 0.90 | 90.13 ± 1.24 | 34.6 |

## Final-B, 5 models × 3 seeds (test 432 images, best.pth)

| Model | Seeds | Test acc (%) | Test macro-F1 (%) | Val acc (%) | Best epoch per seed |
|---|---:|---:|---:|---:|---|
| convnext_tiny | 3 | 98.92 ± 0.27 | 98.99 ± 0.29 | 99.63 ± 0.43 | 18, 20, 10 |
| swin_tiny | 3 | 99.69 ± 0.13 | 99.72 ± 0.12 | 99.67 ± 0.26 | 44, 40, 22 |
| efficientnet_b3 | 3 | 97.92 ± 0.46 | 98.07 ± 0.41 | 98.30 ± 0.59 | 31, 43, 30 |
| resnet50 | 3 | 97.53 ± 0.81 | 97.67 ± 0.82 | 98.30 ± 0.19 | 39, 49, 49 |
| mobilenetv3_large | 3 | 98.53 ± 0.27 | 98.64 ± 0.27 | 98.51 ± 0.33 | 46, 46, 29 |

50 checkpoints listed in `checkpoints.sha256`.
