# Woodbench Timeline

## 2026

| Date | Time | Title | Result | Details |
|------|------|-------|--------|---------|
| 12-05-2026 | 13:41 | Phase 1 code review and gap analysis | ~70% complete | [Details](timeline/12-05-2026.md) |
| 12-05-2026 | 15:50 | Refactor Phase 1 notebook for Colab portability | Done | [Details](timeline/12-05-2026.md) |
| 14-05-2026 | 10:15 | Create EfficientNet-B3 notebook for Phase 1 | Done | [Details](timeline/14-05-2026.md) |
| 14-05-2026 | 12:29 | Update all deprecated PyTorch 2.x APIs across codebase | Done | [Details](timeline/14-05-2026.md) |
| 17-05-2026 | 12:18 | Recalculate batch sizes for Colab A100 40GB (not 80GB) | Done | [Details](timeline/17-05-2026.md) |
| 19-05-2026 | 13:25 | Phase 1 Task A: Complete 5 models (seed=42) | Done | [Details](timeline/19-05-2026.md) |
| 19-05-2026 | 13:25 | Phase 1 Task A: Multi-seed training (seeds 123, 456) | Done | [Details](timeline/19-05-2026.md) |
| 21-05-2026 | 13:00 | Phase 1 Task A: Implementation verification | Done | [Details](timeline/21-05-2026.md) |
| 22-05-2026 | 11:30 | Task B planning + lich 2x4090 (5x3 runs) | ~6.25h wall-clock | [Details](timeline/22-05-2026.md) |
| 22-05-2026 | 12:30 | Task B: notebooks cho 2 machine vast.ai + chiến thuật tối ưu | Done | Machine A: ConvNeXt+MobileNet+EfficientNet; Machine B: Swin+ResNet |
| 22-05-2026 | 13:44 | Task B: 5 notebook đơn giản (1 model × 3 seeds), rollback parallel | Done | 5 notebooks: ConvNeXt, Swin, EfficientNet, MobileNet, ResNet; mỗi notebook 3 seeds tuần tự |
| 23-05-2026 | 01:45 | Task B: fix notebooks cho papermill trên vast.ai | Done | num_workers=0 tránh multiprocessing crash; .env dotenv thay getpass |
| 23-05-2026 | 01:53 | Task B: wave runner script với auto-detect kernel | Done | `scripts/run_taskB_waves.py --all` chạy 3 waves + driver tự động |
| 23-05-2026 | 06:20 | Task B: train.py — thêm test evaluation + đầy đủ metrics wandb | Done | Log val precision/recall/f1 per epoch; test set evaluation sau training; per-class F1; save JSON |
| 24-05-2026 | 14:30 | Task B: Hoàn thành Main Experiments (15 runs) | Done | [Details](timeline/24-05-2026.md) |
