# Task 15 analysis: 14-species runs (07-11-2026)

Script: `notebooks/phase 2/supervisor-feedback/task_15_clean14_runs/analyze_clean14.py`. It reads:

- the 50 runs downloaded from `drive:wood/2026-11-07-clean14-runs/` and extracted into `data/experiments/`
  (zip md5 and all 50 `best.pth` sha256 checked);
- CV v2 (`../../task_01_fivefold_cv_training/results-v2/`);
- the August Final-B registry (`docs/reports/experiments/phase_1b/final_b_new_recomputed/20260819T043733Z/`).

The summary tables are in `../results/summary.md`. This folder adds the checks and comparisons below.

## 1. Integrity

`integrity.csv` covers all 50 runs:

- The accuracy recomputed from the prediction files equals `metrics.json`.
- Every `best.pth` matches the sha256 recorded at the end of its run.
- All runs used code commit `a496be4`, torch 2.14.1+cu126 and the same data checksum list.

## 2. The 50-epoch cap (`convergence.csv`)

27 of 50 runs reached epoch 50, mostly EfficientNet, MobileNet and ResNet. This is the end of the schedule, not a
budget cut in the middle of training:

- With 5 warmup epochs + cosine over 50 epochs, the LR is 4.3e-6 at epoch 45 and close to 0 at epoch 50.
- Early stopping (patience 15 on val loss) rarely fires, because val loss keeps falling slowly while the LR
  anneals.
- From epoch 40 to the best epoch, val loss fell by 0 to 0.019 (label-smoothed CE). Val accuracy moved by −0.3 to
  +0.8 points, i.e. at most about 7 of the 803–945 val images.

The only visible trend is MobileNet in CV. Its val accuracy (mean of 5 folds) was 95.68% at epoch 40, 96.10% at 45
and 96.12% at 50 (`curves.png`).

Running more epochs with the same schedule would not continue this trend. Whether a longer schedule (for
example 100 epochs) helps the slower models is untested. The statement in handoff section 9 that these numbers
are "limited by the maximum number of epochs" is too strong.

## 3. Paired tests (`paired_tests.csv`)

All tests are exact McNemar tests on identical images. They assume images are independent, but images of one
origin (specimen) are not. When the errors cluster in a few origins, the p-values are too small. Read them as
"is the difference more than a handful of images", not as formal evidence.

### 3.1 CV, 14-species models vs CV v2 (17-species models)

Same 4,115 out-of-fold images, same folds:

| Model | CV v2 | 14 species | Only new right / only old right | p |
|---|---:|---:|---:|---:|
| ConvNeXt-Tiny | 99.03 | 99.44 | 24 / 7 | 0.003 |
| Swin-Tiny | 98.88 | 99.34 | 35 / 16 | 0.011 |
| EfficientNet-B3 | 97.47 | 97.57 | 44 / 40 | 0.74 |
| ResNet-50 | 96.89 | 97.13 | 43 / 33 | 0.30 |
| MobileNetV3-Large | 96.23 | 95.29 | 65 / 104 | 0.003 |

Two things changed at once: 3 species were removed, and the checkpoint rule changed (val acc from epoch 1 with
patience 10, versus val loss after warmup with patience 15). These runs cannot separate the two effects.

### 3.2 Model ranking

CV OOF:

- ConvNeXt and Swin are tied: 99.44 vs 99.34; 13 vs 9 discordant images; p = 0.52.
- Both are clearly above EfficientNet (+81/−4).
- EfficientNet and ResNet are tied (p = 0.11).
- MobileNet is last.

Final-B, pooled over seeds: Swin 99.69 vs ConvNeXt 98.92 (11 vs 1 discordant image-runs, p = 0.006). The Final-B
test set makes this difference fragile (section 4).

## 4. The Final-B test split holds only 14 specimens

The fixed split has, for the 14 species:

| Split | Images | Origins (specimens) |
|---|---:|---:|
| Train | 2,880 | 98 |
| Val | 803 | 28 |
| Test | 432 | 14 (one per species) |

Over the 15 Final-B runs there are 96 misclassified image-runs (`final_b_test_errors.csv`). Of these, **66 (69%)
come from one specimen: Pinus krempfii 3384.4, photos 128–136.** Photo 130 is wrong in 15/15 runs, and photos
129, 131 and 133 in 10/15. These are the same photos as in task 14 (August reproduction). By model:

| Model | Test errors, 3 seeds | Errors on 3384.4 |
|---|---:|---:|
| Swin | 4 | 4 |
| ConvNeXt | 14 | 6 |
| MobileNet | 19 | 14 |
| EfficientNet | 27 | 20 |
| ResNet | 32 | 22 |

Swin's lead on Final-B therefore comes down to how the models handle one krempfii specimen, plus ConvNeXt's 8
errors on Taxus 3391.1 and Podocarpus 3386.1. CV spreads the 140 origins over 5 held-out folds, and there
ConvNeXt and Swin tie.

**Recommendation for the paper:** use CV (140 specimens) as the main comparison, and Final-B (14 test specimens)
as the fixed-split check with this caveat.

## 5. August Final-B (17 species) vs 14-species Final-B (`august_vs_clean14.csv`)

Both on the same 432 test images. For the August runs, a prediction of one of the 3 removed species counts as
wrong. Values are mean over seeds 42/123/456, in %:

| Model | August | 14 species | p (pooled McNemar) |
|---|---:|---:|---:|
| ConvNeXt-Tiny | 99.46 | 98.92 | 0.12 |
| Swin-Tiny | 98.92 | 99.69 | 0.002 |
| EfficientNet-B3 | 97.61 | 97.92 | 0.57 |
| ResNet-50 | 91.44 | 97.53 | 1e-16 |
| MobileNetV3-Large | 98.15 | 98.53 | 0.36 |

The ResNet jump is the LR fix (task 14 / handoff 01-11 section 3.1). Under the August substring rule, ResNet's
whole backbone ran at 1e-5, while ConvNeXt/Swin effectively ran at about 1e-4. The ConvNeXt drop of 0.54 points
is about 2 images per seed and is not significant.

## 6. Where the errors are

CV OOF errors, 14-species RGB, 5 models (`oof_errors_by_origin.csv`, `confusions.csv`):

- 462 errors in total, in 66 of 140 origins. The top 5 origins hold 38%.
- Worst origins:
  - Pinus dalatensis 3393.1: 47 errors over 5 × 28 image-runs. Task 11 already flagged this specimen.
  - Pinus cernua 3385.10: 41 errors, 40 of them by EfficientNet, ResNet and MobileNet.
  - P. dalatensis 3393.6 and Taxus 3391.9: 33 and 28 errors.
  - P. krempfii 3384.4: 28 errors.
- Confusion pairs:
  - P. dalatensis ↔ P. krempfii: 81 + 40 in CV; in Final-B test, krempfii → dalatensis 66 times.
  - Taxus wallichiana ↔ Podocarpus pilgeri: 45 + 18.
  - Pinus cernua → dalatensis / Pseudotsuga: 28 / 22.

## 7. Grayscale and 16 px: which species lose (`degraded_per_species.csv`)

ConvNeXt OOF F1 per species (5 folds):

| Species | RGB | Grayscale | 16 px |
|---|---:|---:|---:|
| Taxus wallichiana | 0.998 | 0.812 | 0.761 |
| Podocarpus pilgeri | 1.000 | 0.867 | 0.760 |
| Xanthocyparis vietnamensis | 0.996 | 0.887 | 0.861 |
| Pinus dalatensis | 0.970 | 0.877 | 0.852 |
| Pinus krempfii | 0.974 | 0.913 | 0.890 |
| Pinus cernua | 0.991 | 0.979 | 0.815 |
| 8 other species | 1.000 | ≥ 0.956 | ≥ 0.893 |

Grayscale hurts mostly Taxus and Podocarpus (already the second most confused pair in RGB) and Xanthocyparis. Their
separation relies on colour. 16 px also removes Pinus cernua's cue (0.979 in grayscale → 0.815). Most species stay
≥ 0.95 even at 16 px, so the model relies a lot on low-frequency colour and texture, not only on fine anatomy.

In grayscale, accuracy rises with the fold index (91.8 → 97.2). The per-species table does not explain this; it
is probably which specimens land in which fold. With 5 folds it should not be interpreted.

## 8. `curves.png`

Mean val loss and val accuracy per epoch: CV over 5 folds, Final-B over 3 seeds. ConvNeXt and Swin plateau by
about epoch 15–25. EfficientNet and ResNet plateau by about 35–40. MobileNet in CV flattens out after about epoch 45.
