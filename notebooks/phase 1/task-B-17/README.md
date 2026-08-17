# Final Task B 17-class

This folder contains the final Task B species-classification notebooks for the
17-class dataset.

Run order:

1. `00_data_preparation.ipynb` if `data/task-B-group-split` still contains the
   3 excluded classes.
2. One model notebook at a time:
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

Outputs are written under:

```text
data/experiments/Task-B-17class-scratch/
```

These runs start from ImageNet pretrained weights and do not transfer weights
from the previous 20-class Task B checkpoints.

The metrics notebook reads the finalized checkpoints from
`data/experiments/Task-B/` and writes recomputed artifacts to
`data/experiments/phase_1/final_b_recomputed/`.
