# Plan: 5-Fold CV by Origin Key

## Context
Supervisor yêu cầu 5-fold CV theo origin_key để đánh giá true generalization. Hiện tại 14/17 species chỉ có 1 origin trong test set, khiến confidence intervals không reliable.

## Requirements
1. **Retrain all 5 models** × 5 folds = 25 runs
2. **GroupKFold by origin_key** thay vì random split
3. **WandB logging** cho mỗi run
4. **Scripts thay vì notebook** để chạy trên vast.ai
5. **Chạy ngầm** (nohup/screen) vì SSH có thể mất kết nối

## Scripts Structure

```
notebooks/phase 2/supervisor-feedback/task_01_fivefold_cv_training/
├── train_kfold.py          # Main training script
├── train_single_fold.py    # Train 1 fold (entry point per model)
├── kfold_cv.sh             # Launcher script for all 5 models
├── run_model.sh            # Run 1 model × 5 folds
├── run_fold.sh             # Run 1 fold
├── requirements.txt
└── README.md
```

## Key Differences from Original

| Aspect | Original (Random Split) | New (Origin-Key Split) |
|--------|------------------------|------------------------|
| Split method | `train_test_split` random | `GroupKFold` by origin_key |
| Validation | Random samples | All samples from unseen origins |
| Purpose | Optimize hyperparameters | Evaluate true generalization |
| Metric | Accuracy only | Accuracy + confidence intervals |

## Implementation

### 1. Extract Origin Key
```python
def extract_origin_key(filename):
    # Format: "3351. Araucaria araucana.1.(12).png"
    # Origin: "3351. Araucaria araucana.1"
    if ' (' in filename:
        return filename.rsplit(' (', 1)[0]
    return filename.rsplit('.png', 1)[0] if '.png' in filename else filename
```

### 2. GroupKFold Split
```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, groups=origin_keys)):
    # train_idx: samples from train_origins
    # val_idx: samples from val_origins (unseen)
```

### 3. Dataset with Origin Key
```python
class OriginKFoldDataset(Dataset):
    def __init__(self, samples, labels, origin_keys, transform):
        # samples, labels, origin_keys aligned
```

## Output Structure

```
data/experiments/Task-B-origin-cv/
├── convnext_tiny/
│   ├── fold_1/
│   │   ├── best.pth
│   │   ├── last.pth
│   │   └── metrics.json
│   ├── fold_2/
│   │   └── ...
│   ├── fold_3/
│   ├── fold_4/
│   ├── fold_5/
│   └── summary.csv  (mean ± std across folds)
├── mobilenetv3_large/
│   └── ...
├── resnet50/
│   └── ...
├── efficientnet_b3/
│   └── ...
└── swin_tiny/
    └── ...
```

## WandB Project
- Project: `woodbench-origin-cv`
- Runs: `{backbone}_fold_{i}` e.g., `convnext_tiny_fold_1`

## Time Estimate

| Model | Time per fold | 5 folds | Notes |
|-------|--------------|--------|-------|
| ConvNeXt-Tiny | ~30 min | ~2.5h | Best performer |
| Swin-Tiny | ~35 min | ~3h | Complex architecture |
| MobileNetV3-Large | ~20 min | ~1.7h | Lightweight |
| EfficientNet-B3 | ~25 min | ~2h | Medium |
| ResNet-50 | ~25 min | ~2h | Baseline |

**Total: ~11-12 hours** for all 5 models × 5 folds

## Verification

1. Check `origin_key` extraction correctness
2. Verify no origin leakage between train/val
3. Compare accuracy vs random split
4. Generate confidence intervals from 5 folds

## Usage on vast.ai

```bash
# SSH to vast.ai instance
# Activate conda environment
conda activate wood

# Copy scripts
rsync -avz scripts/ user@vast:/workspace/woodbench/

# Run all models in parallel
nohup bash kfold_cv.sh > kfold.log 2>&1 &

# Monitor
tail -f kfold.log
# or
tensorboard --logdir=data/experiments/Task-B-origin-cv/
```
