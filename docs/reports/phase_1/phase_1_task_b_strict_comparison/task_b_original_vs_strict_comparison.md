# Task B Original vs Strict-by-Origin Comparison

- Generated: 2026-06-27 11:40:53
- Original split source: `docs/reports/phase_1/phase_1_run_report_not_strict [DEPRECATED]/phase1_summary_by_group_backbone_percent.csv`.
- Strict split source: `docs/reports/phase_1/task_b_wandb_all_runs_report-hotfix.md` plus local seed-42 supplement noted in that file.

## Key Findings

- Best strict Task B main backbone: `swin_tiny` with mean F1 99.41% across 3 seeds.
- Largest F1 drop from original to strict split: `resnet50` (-11.56 percentage points).
- Strict-by-origin split is materially harder for weaker/less robust backbones, especially ResNet-50, EfficientNet-B3, and MobileNetV3-Large.

## Main Comparison

| Backbone        | Original Acc | Strict Acc | Delta Acc | Original F1 | Strict F1 | Delta F1 |
| --------------- | ------------ | ---------- | --------- | ----------- | --------- | -------- |
| ConvNeXt-Tiny   | 99.68        | 98.61      | -1.07     | 99.68       | 98.9      | -0.78    |
| Swin-Tiny       | 99.46        | 99.5       | 0.04      | 99.46       | 99.41     | -0.05    |
| EfficientNet-B3 | 98.52        | 97.35      | -1.17     | 98.56       | 92.14     | -6.42    |
| MobileNetV3-L   | 98.63        | 92.94      | -5.69     | 98.61       | 89.03     | -9.58    |
| ResNet-50       | 93.76        | 84.57      | -9.19     | 93.77       | 82.21     | -11.56   |
