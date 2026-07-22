# Task-B W&B Runs Report

- Source: W&B `ny0g507h3p_/woodbench-taskB-clean`
- Local supplement: `data/experiments/Task-B/main/seed_42/*/test_metrics.json`
- Extracted at: 2026-06-27 03:06:55 UTC
- Total W&B runs in project: 54
- Scope note: this report intentionally includes all runs found in the W&B project, including main training, eval-only runs, confusion-matrix runs, ablation runs, and empty/stub runs.

## Category Counts

| Category | Runs | Runs with test metric |
|---|---:|---:|
| M1_main / confusion-matrix extra | 5 | 0 |
| M1_main / test-eval extra | 18 | 18 |
| M1_main / training | 15 | 15 |
| M2_ablation / 3.1_90rotation | 5 | 5 |
| M2_ablation / 3.2_label_smoothing | 10 | 10 |
| Other / unclassified | 1 | 0 |

## Quick Highlights

- Best test accuracy among all metric-bearing runs: `convnext_tiny_90rot_s42` (100.00%, run `k95af1nw`).
- Best M1 training run by test accuracy: `swin_tiny_s123` (99.75%).
- Best M2 ablation run by test accuracy: `convnext_tiny_90rot_s42` (100.00%).

## M1_main / training

| Run | ID | Created | State | Backbone | Seed | Variant | LS | Val Acc (best/final) | Best Epoch | Epochs | Early Stop | Test Acc | Test F1 | Test Prec | Test Rec | Summary Keys |
|---|---|---|---|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| [mobilenetv3_large_s42](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/yt9cfgkp) | `yt9cfgkp` | 2026-06-25T07:46:16Z | finished | mobilenetv3_large | 42 | main training | - | 96.01% | 36 | 36 | False | 92.94% | 88.92% | 87.82% | 91.35% | 9 |
| [swin_tiny_s42](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/6kshh9tw) | `6kshh9tw` | 2026-06-25T07:46:21Z | finished | swin_tiny | 42 | main training | - | 100.00% | 30 | 30 | False | 99.12% | 98.95% | 98.93% | 99.07% | 9 |
| [resnet50_s42](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/fdfa6kkg) | `fdfa6kkg` | 2026-06-25T07:46:27Z | finished | resnet50 | 42 | main training | - | 91.47% | 45 | 45 | False | 84.62% | 81.88% | 83.02% | 85.35% | 9 |
| [efficientnet_b3_s42](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/9vkvbu97) | `9vkvbu97` | 2026-06-25T07:46:34Z | finished | efficientnet_b3 | 42 | main training | - | 96.35% | 45 | 45 | False | 97.73% | 92.68% | 92.93% | 92.63% | 9 |
| [convnext_tiny_s42](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/d76qd4wl) | `d76qd4wl` | 2026-06-25T07:46:48Z | finished | convnext_tiny | 42 | main training | - | 100.00% | 20 | 20 | False | 98.99% | 98.94% | 98.79% | 99.15% | 9 |
| [resnet50_s123](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/3re55dns) | `3re55dns` | 2026-06-25T12:00:01Z | finished | resnet50 | 123 | main training | - | 91.81% | 42 | 50 | False | 84.11% | 82.22% | 84.36% | 85.59% | 17 |
| [swin_tiny_s123](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/q45b80ny) | `q45b80ny` | 2026-06-25T12:00:21Z | finished | swin_tiny | 123 | main training | - | 99.89% | 21 | 31 | True | 99.75% | 99.72% | 99.64% | 99.80% | 17 |
| [mobilenetv3_large_s123](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/usec23m0) | `usec23m0` | 2026-06-25T12:00:56Z | finished | mobilenetv3_large | 123 | main training | - | 95.79% | 40 | 50 | True | 92.06% | 88.37% | 87.56% | 91.32% | 17 |
| [convnext_tiny_s123](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/vitm4mwe) | `vitm4mwe` | 2026-06-25T12:01:10Z | finished | convnext_tiny | 123 | main training | - | 99.56% | 25 | 35 | True | 98.87% | 99.17% | 98.84% | 99.57% | 17 |
| [efficientnet_b3_s123](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/pc1cbyug) | `pc1cbyug` | 2026-06-25T12:01:28Z | finished | efficientnet_b3 | 123 | main training | - | 95.68% | 26 | 36 | True | 97.60% | 92.18% | 92.51% | 92.24% | 17 |
| [swin_tiny_s456](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/wxxkmrlj) | `wxxkmrlj` | 2026-06-25T14:16:50Z | finished | swin_tiny | 456 | main training | - | 99.78% | 18 | 28 | True | 99.62% | 99.57% | 99.50% | 99.66% | 17 |
| [convnext_tiny_s456](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/a9xc19m2) | `a9xc19m2` | 2026-06-25T14:33:39Z | finished | convnext_tiny | 456 | main training | - | 99.34% | 21 | 31 | True | 97.98% | 98.58% | 98.18% | 99.22% | 17 |
| [efficientnet_b3_s456](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/5dbl0ta3) | `5dbl0ta3` | 2026-06-25T15:04:10Z | finished | efficientnet_b3 | 456 | main training | - | 95.90% | 29 | 39 | True | 96.72% | 91.57% | 91.66% | 91.98% | 17 |
| [resnet50_s456](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/neutg2dc) | `neutg2dc` | 2026-06-25T15:50:59Z | finished | resnet50 | 456 | main training | - | 92.47% | 42 | 50 | False | 84.99% | 82.54% | 84.22% | 86.53% | 17 |
| [mobilenetv3_large_s456](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/74r2kqr1) | `74r2kqr1` | 2026-06-25T15:52:28Z | finished | mobilenetv3_large | 456 | main training | - | 96.79% | 38 | 48 | True | 93.82% | 89.81% | 89.02% | 91.71% | 17 |

## M1_main / test-eval extra

| Run | ID | Created | State | Backbone | Seed | Variant | LS | Val Acc (best/final) | Best Epoch | Epochs | Early Stop | Test Acc | Test F1 | Test Prec | Test Rec | Summary Keys |
|---|---|---|---|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| [swin_tiny_s42_test](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/7svlabs7) | `7svlabs7` | 2026-06-25T16:27:21Z | finished | swin_tiny | 42 | test eval | - | - | - | - | - | 99.12% | 98.95% | 98.93% | 99.07% | 27 |
| [swin_tiny_s123_test](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/67m0auuj) | `67m0auuj` | 2026-06-25T16:28:09Z | finished | swin_tiny | 123 | test eval | - | - | - | - | - | 98.87% | 98.83% | 98.49% | 99.26% | 27 |
| [swin_tiny_s456_test](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/h7rgo672) | `h7rgo672` | 2026-06-25T16:28:58Z | finished | swin_tiny | 456 | test eval | - | - | - | - | - | 99.75% | 99.72% | 99.65% | 99.80% | 27 |
| [swin_tiny_s42_test](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/wmqbyuyl) | `wmqbyuyl` | 2026-06-25T17:00:10Z | finished | swin_tiny | 42 | test eval | - | - | - | - | - | 99.12% | 98.95% | 98.93% | 99.07% | 28 |
| [convnext_tiny_s42_test](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/rd4unl55) | `rd4unl55` | 2026-06-25T17:01:04Z | finished | convnext_tiny | 42 | test eval | - | - | - | - | - | 98.99% | 98.94% | 98.79% | 99.15% | 28 |
| [swin_tiny_s123_test](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/khth3mer) | `khth3mer` | 2026-06-25T17:01:07Z | finished | swin_tiny | 123 | test eval | - | - | - | - | - | 98.87% | 98.83% | 98.49% | 99.26% | 28 |
| [convnext_tiny_s123_test](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/6ool5o45) | `6ool5o45` | 2026-06-25T17:01:57Z | finished | convnext_tiny | 123 | test eval | - | - | - | - | - | 98.99% | 99.24% | 98.94% | 99.60% | 28 |
| [swin_tiny_s456_test](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/pcx1b536) | `pcx1b536` | 2026-06-25T17:02:01Z | finished | swin_tiny | 456 | test eval | - | - | - | - | - | 99.75% | 99.72% | 99.65% | 99.80% | 28 |
| [convnext_tiny_s456_test](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/tr6a33q7) | `tr6a33q7` | 2026-06-25T17:02:35Z | finished | convnext_tiny | 456 | test eval | - | - | - | - | - | 97.10% | 98.13% | 97.71% | 98.98% | 28 |
| [efficientnet_b3_s42_test](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/pk21hk9q) | `pk21hk9q` | 2026-06-25T18:03:05Z | finished | efficientnet_b3 | 42 | test eval | - | - | - | - | - | 97.73% | 92.68% | 92.93% | 92.63% | 28 |
| [efficientnet_b3_s123_test](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/ajiwupvl) | `ajiwupvl` | 2026-06-25T18:03:42Z | finished | efficientnet_b3 | 123 | test eval | - | - | - | - | - | 96.97% | 91.61% | 91.55% | 91.99% | 28 |
| [efficientnet_b3_s456_test](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/6qc0oz2d) | `6qc0oz2d` | 2026-06-25T18:04:15Z | finished | efficientnet_b3 | 456 | test eval | - | - | - | - | - | 96.60% | 91.62% | 91.76% | 92.09% | 28 |
| [resnet50_s42_test](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/dd703thy) | `dd703thy` | 2026-06-25T18:19:10Z | finished | resnet50 | 42 | test eval | - | - | - | - | - | 84.62% | 81.88% | 83.02% | 85.35% | 28 |
| [mobilenetv3_large_s42_test](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/3b0zq1ht) | `3b0zq1ht` | 2026-06-25T18:19:22Z | finished | mobilenetv3_large | 42 | test eval | - | - | - | - | - | 92.94% | 88.92% | 87.82% | 91.35% | 28 |
| [resnet50_s123_test](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/euwf4inu) | `euwf4inu` | 2026-06-25T18:19:40Z | finished | resnet50 | 123 | test eval | - | - | - | - | - | 84.62% | 83.14% | 86.10% | 86.22% | 28 |
| [mobilenetv3_large_s123_test](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/mrfm96z8) | `mrfm96z8` | 2026-06-25T18:19:55Z | finished | mobilenetv3_large | 123 | test eval | - | - | - | - | - | 91.55% | 87.86% | 87.04% | 90.99% | 28 |
| [resnet50_s456_test](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/h9crg3ri) | `h9crg3ri` | 2026-06-25T18:20:11Z | finished | resnet50 | 456 | test eval | - | - | - | - | - | 85.50% | 82.95% | 84.33% | 86.58% | 28 |
| [mobilenetv3_large_s456_test](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/iaelf2ak) | `iaelf2ak` | 2026-06-25T18:20:27Z | finished | mobilenetv3_large | 456 | test eval | - | - | - | - | - | 93.57% | 89.64% | 88.88% | 91.61% | 28 |

## M1_main / confusion-matrix extra

| Run | ID | Created | State | Backbone | Seed | Variant | LS | Val Acc (best/final) | Best Epoch | Epochs | Early Stop | Test Acc | Test F1 | Test Prec | Test Rec | Summary Keys |
|---|---|---|---|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| [swin_tiny_CM](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/zn0svldw) | `zn0svldw` | 2026-06-25T17:02:10Z | finished | swin_tiny | - | confusion matrix | - | - | - | - | - | - | - | - | - | 5 |
| [convnext_tiny_CM](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/rq6c4ona) | `rq6c4ona` | 2026-06-25T17:02:43Z | finished | convnext_tiny | - | confusion matrix | - | - | - | - | - | - | - | - | - | 5 |
| [efficientnet_b3_CM](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/x11s11oy) | `x11s11oy` | 2026-06-25T18:04:21Z | finished | efficientnet_b3 | - | confusion matrix | - | - | - | - | - | - | - | - | - | 5 |
| [resnet50_CM](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/7uqpsk95) | `7uqpsk95` | 2026-06-25T18:20:18Z | finished | resnet50 | - | confusion matrix | - | - | - | - | - | - | - | - | - | 5 |
| [mobilenetv3_large_CM](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/1p5g473u) | `1p5g473u` | 2026-06-25T18:20:34Z | finished | mobilenetv3_large | - | confusion matrix | - | - | - | - | - | - | - | - | - | 5 |

## M2_ablation / 3.1_90rotation

| Run | ID | Created | State | Backbone | Seed | Variant | LS | Val Acc (best/final) | Best Epoch | Epochs | Early Stop | Test Acc | Test F1 | Test Prec | Test Rec | Summary Keys |
|---|---|---|---|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| [convnext_tiny_90rot_s42](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/k95af1nw) | `k95af1nw` | 2026-06-23T13:40:49Z | finished | convnext_tiny | 42 | 90 rotation | - | 99.67% | - | - | - | 100.00% | 100.00% | 100.00% | 100.00% | 32 |
| [swin_tiny_90rot_s42](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/1chj6m9n) | `1chj6m9n` | 2026-06-23T13:50:24Z | finished | swin_tiny | 42 | 90 rotation | - | 99.34% | - | - | - | 99.87% | 99.82% | 99.84% | 99.80% | 32 |
| [efficientnet_b3_90rot_s42](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/4c244k0f) | `4c244k0f` | 2026-06-23T17:19:06Z | finished | efficientnet_b3 | 42 | 90 rotation | - | 96.90% | - | - | - | 98.87% | 98.51% | 98.68% | 98.50% | 32 |
| [mobilenetv3_large_90rot_s42](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/lkc3t81r) | `lkc3t81r` | 2026-06-23T17:24:22Z | finished | mobilenetv3_large | 42 | 90 rotation | - | 98.01% | - | - | - | 99.12% | 98.91% | 98.95% | 98.96% | 32 |
| [resnet50_90rot_s42](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/m2ierzfb) | `m2ierzfb` | 2026-06-23T18:50:11Z | finished | resnet50 | 42 | 90 rotation | - | 93.47% | - | - | - | 96.47% | 95.24% | 95.59% | 95.46% | 32 |

## M2_ablation / 3.2_label_smoothing

| Run | ID | Created | State | Backbone | Seed | Variant | LS | Val Acc (best/final) | Best Epoch | Epochs | Early Stop | Test Acc | Test F1 | Test Prec | Test Rec | Summary Keys |
|---|---|---|---|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| [convnext_tiny_ls0.0_s42](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/cpmnzk47) | `cpmnzk47` | 2026-06-24T03:29:51Z | finished | convnext_tiny | 42 | label smoothing | 0.0 | 100.00% | 21 | 31 | True | 98.23% | 97.83% | 97.98% | 98.15% | 17 |
| [efficientnet_b3_ls0.0_s42](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/c6w77bft) | `c6w77bft` | 2026-06-24T03:31:44Z | finished | efficientnet_b3 | 42 | label smoothing | 0.0 | 95.79% | 45 | 50 | False | 96.72% | 91.52% | 91.58% | 91.78% | 17 |
| [mobilenetv3_large_ls0.0_s42](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/rh0e0g54) | `rh0e0g54` | 2026-06-24T03:33:21Z | finished | mobilenetv3_large | 42 | label smoothing | 0.0 | 97.34% | 35 | 45 | True | 94.45% | 90.43% | 89.71% | 92.00% | 17 |
| [convnext_tiny_ls0.2_s42](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/z08vm1tt) | `z08vm1tt` | 2026-06-24T04:24:30Z | finished | convnext_tiny | 42 | label smoothing | 0.2 | 99.89% | 30 | 40 | True | 99.50% | 99.60% | 99.46% | 99.75% | 17 |
| [mobilenetv3_large_ls0.2_s42](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/e0taqezl) | `e0taqezl` | 2026-06-24T04:54:17Z | finished | mobilenetv3_large | 42 | label smoothing | 0.2 | 97.01% | 25 | 35 | True | 95.33% | 90.91% | 89.94% | 92.38% | 17 |
| [efficientnet_b3_ls0.2_s42](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/5tlpar9u) | `5tlpar9u` | 2026-06-24T05:02:26Z | finished | efficientnet_b3 | 42 | label smoothing | 0.2 | 95.90% | 33 | 43 | True | 97.73% | 92.52% | 92.77% | 92.49% | 17 |
| [resnet50_ls0.0_s42](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/65nvbbtf) | `65nvbbtf` | 2026-06-24T06:42:18Z | finished | resnet50 | 42 | label smoothing | 0.0 | 92.91% | 49 | 50 | False | 87.77% | 84.65% | 85.40% | 87.50% | 17 |
| [swin_tiny_ls0.0_s42](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/8mlqzuqj) | `8mlqzuqj` | 2026-06-24T06:42:26Z | finished | swin_tiny | 42 | label smoothing | 0.0 | 99.78% | 11 | 21 | True | 98.49% | 98.03% | 98.18% | 98.14% | 17 |
| [swin_tiny_ls0.2_s42](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/vg4zfqu8) | `vg4zfqu8` | 2026-06-24T07:14:11Z | finished | swin_tiny | 42 | label smoothing | 0.2 | 99.78% | 14 | 24 | True | 99.87% | 99.84% | 99.84% | 99.85% | 17 |
| [resnet50_ls0.2_s42](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/we0so9af) | `we0so9af` | 2026-06-24T07:58:39Z | finished | resnet50 | 42 | label smoothing | 0.2 | 92.36% | 28 | 38 | True | 88.40% | 84.79% | 84.57% | 87.65% | 17 |

## Other / unclassified

| Run | ID | Created | State | Backbone | Seed | Variant | LS | Val Acc (best/final) | Best Epoch | Epochs | Early Stop | Test Acc | Test F1 | Test Prec | Test Rec | Summary Keys |
|---|---|---|---|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| [efficiency_benchmark](https://wandb.ai/ny0g507h3p_/woodbench-taskB-clean/runs/7rgrh9lx) | `7rgrh9lx` | 2026-06-25T18:44:51Z | finished | - | - | - | - | - | - | - | - | - | - | - | - | 5 |

## M1 Training Aggregate By Backbone

| Backbone | Runs with Test Acc | Mean Test Acc | Best Test Acc | Best Run | Mean Test F1 |
|---|---:|---:|---:|---|---:|
| convnext_tiny | 3 | 98.61% | 98.99% | `convnext_tiny_s42` | 98.90% |
| efficientnet_b3 | 3 | 97.35% | 97.73% | `efficientnet_b3_s42` | 92.14% |
| mobilenetv3_large | 3 | 92.94% | 93.82% | `mobilenetv3_large_s456` | 89.03% |
| resnet50 | 3 | 84.57% | 84.99% | `resnet50_s456` | 82.21% |
| swin_tiny | 3 | 99.50% | 99.75% | `swin_tiny_s123` | 99.41% |

## Notes

- Runs named `*_s42` in `M1_main / training` had validation-only W&B summaries online; missing best-epoch/test metrics were filled from local `data/experiments/Task-B/main/seed_42/*/test_metrics.json`.
- Runs named `*_CM` are confusion-matrix/media logging runs, not training runs.
- Metrics are read from W&B run summaries except the seed-42 local backfill noted above. For validation accuracy, the report uses `best_val_acc` when available and falls back to `val/acc`; for slash metric names (`test/accuracy`, `test/f1`, etc.), the report normalizes them into the Test columns.
- No legacy CSV reports were used to populate this file.
