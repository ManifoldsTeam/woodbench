| Model | Backbone LR | Val F1 | Val acc | Val loss | Best epoch | Epochs | Test acc | Test F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| convnext_tiny | 1e-05 | 99.85 | 99.89 | 0.6052 | 10 | 20 | 98.68 | 98.79 |
| convnext_tiny | 0.0001 | 99.68 | 99.67 | 0.5867 | 12 | 22 | 99.44 | 99.44 |
| efficientnet_b3 | 1e-05 | 95.45 | 95.68 | 0.7065 | 31 | 41 | 95.68 | 95.58 |
| efficientnet_b3 | 0.0001 | 98.44 | 98.45 | 0.6243 | 17 | 27 | 98.31 | 98.40 |
| mobilenetv3_large | 1e-05 | 93.44 | 93.69 | 0.7768 | 37 | 47 | 97.74 | 97.74 |
| mobilenetv3_large | 0.0001 | 98.66 | 98.67 | 0.6182 | 34 | 44 | 98.87 | 98.96 |
| resnet50 | 1e-05 | 89.27 | 90.37 | 0.9111 | 41 | 50 | 93.05 | 93.33 |
| resnet50 | 0.0001 | 98.69 | 98.67 | 0.6229 | 31 | 41 | 98.12 | 98.21 |
| swin_tiny | 1e-05 | 99.81 | 99.78 | 0.5989 | 18 | 28 | 99.25 | 99.27 |
| swin_tiny | 0.0001 | 100.00 | 100.00 | 0.5759 | 26 | 36 | 99.62 | 99.66 |

| Model | Chosen backbone LR | Reason |
|---|---:|---|
| convnext_tiny | 0.0001 | tie on val macro-F1 (0.17 pt < 0.5); lower val loss (0.5867) |
| efficientnet_b3 | 0.0001 | val macro-F1 higher by 2.99 pt |
| mobilenetv3_large | 0.0001 | val macro-F1 higher by 5.23 pt |
| resnet50 | 0.0001 | val macro-F1 higher by 9.42 pt |
| swin_tiny | 0.0001 | tie on val macro-F1 (0.19 pt < 0.5); lower val loss (0.5759) |

All models choose backbone LR 0.0001: use it for steps 2-4.
