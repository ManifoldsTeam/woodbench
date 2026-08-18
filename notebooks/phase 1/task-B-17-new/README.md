# Task B 17-class — Official Retraining

This folder contains the clean notebooks for the official Task B 17-class
retraining. Historical notebooks and outputs are archived in `../task-B-17-old/`.

Run order:

1. Verify that `data/task-B-group-split` already contains the final 17 classes.
   If preparation is required, run `../task-B-17-old/00_data_preparation.ipynb`.
2. Run one model notebook at a time:
   - `01_train_resnet50.ipynb`
   - `02_train_efficientnet_b3.ipynb`
   - `03_train_mobilenetv3_large.ipynb`
   - `04_train_convnext_tiny.ipynb`
   - `05_train_swin_tiny.ipynb`
3. Recompute all validation/test metrics from the 30 final checkpoints:
   - `06_recompute_all_metrics.ipynb`

Each model notebook runs:

- `main`: seeds 42, 123, 456
- `taskB_90rotation`: seed 42
- `taskB_label_smoothing_0.0`: seed 42
- `taskB_label_smoothing_0.2`: seed 42

Official retraining outputs are written under:

```text
data/experiments/Task-B-new/final/
```

These runs start from ImageNet pretrained weights and do not transfer weights
from the previous 20-class Task B checkpoints.

The metrics notebook reads the new finalized checkpoints from
`data/experiments/Task-B-new/final/` and writes recomputed artifacts to
`data/experiments/phase_1/final_b_new_recomputed/`.

`05_train_swin_tiny_audit.ipynb` preserves the three-run validation audit and
is not part of the official Run All sequence.
