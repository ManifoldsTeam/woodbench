---
phase: planning
title: Conifer Family Classification (Task A) — Project Planning
description: Task breakdown and timeline for family-level classification of 4 conifer families
---

# Project Planning & Task Breakdown — Task A: Conifer Family Classification

## Milestones

- [x] **M0: Dataset Exploration** — Understand raw data structure, species list, image quality
- [x] **M1: Phase 0 Complete** — Dataset organized, labels resolved, split created, notebook ready
- [x] **M2: Baseline Pipeline** — Code structure ready, one model trains successfully
  - ResNet-50 trained on imbalanced data (4,487 samples) → Taxaceae F1 = 70.27%
  - Retrained on balanced data (7,264 samples) → All models >97% accuracy
- [x] **M3: All 5 Models Trained** — Task A complete for all architectures (fine-tuning, differential LR, label_smoothing=0.1, 3 seeds)
  - ConvNeXt-Tiny: 99.90% ± 0.10%
  - Swin-Tiny: 99.80% ± 0.11%
  - EfficientNet-B3: 99.36% ± 0.06%
  - MobileNetV3-L: 99.15% ± 0.06%
  - ResNet-50: 98.46% ± 0.71%
- [ ] **M4: Efficiency Analysis** — Inference time, FLOPs, model size measured (partial: ConvNeXt done ~4ms)
- [ ] **M5: Ablation Complete** — 90° rotation augmentation tested
- [x] **M6: Multi-seed Validation** — 3 seeds run, mean ± std reported
- [ ] **M7: Visualization Complete** — Confusion matrix, Grad-CAM (code ready, generate pending)

---

## Task Breakdown

### Phase -1: Dataset Exploration (NEW — Before Phase 0)

> **Purpose**: Understand raw data before organizing. Not owner of dataset — need to discover what's actually there.

| Task     | Description                | Effort                                | Output |
| -------- | -------------------------- | ------------------------------------- | ------ | ------------------------------------- |
| **-1.1** | Explore raw data structure | List folders, count images per folder | 2h     | Structure documented                  |
| **-1.2** | Verify species list        | Compare against tree.txt references   | 2h     | Species list verified                 |
| **-1.3** | Check image quality        | Resolution, format, corruption check  | 2h     | Quality report                        |
| **-1.4** | Identify ambiguous labels  | Calocedrus sp., Pinus sp., duplicates | 2h     | Ambiguity report                      |
| **-1.5** | Document findings          | Create exploration notebook           | 4h     | `notebooks/dataset_exploration.ipynb` |

### Phase 0: Preparation (Tuần 1-2)

| Task    | Description                                                          | Effort | Dependency |
| ------- | -------------------------------------------------------------------- | ------ | ---------- |
| **0.1** | Organize dataset folder structure `data/{split}/{family}/{species}/` | 1 day  | -1.x       |
| **0.2** | Create `species_to_family.json` label mapping                        | 1 day  | 0.1        |
| **0.3** | Verify species-to-family mapping against exploration findings        | 1 day  | 0.2        |
| **0.4** | Create stratified split 70/15/15, seed=42                            | 1 day  | 0.3        |
| **0.5** | Verify split: check class distribution                               | 1 day  | 0.4        |
| **0.6** | Visualize sample images per species (from exploration findings)      | 1 day  | -1.x       |
| **0.7** | Setup Python environment (PyTorch, timm, etc.)                       | 1 day  | None       |
| **0.8** | Test GPU availability and CUDA setup                                 | 1 day  | 0.7        |
| **0.9** | Profile baseline inference time (dummy model)                        | 1 day  | 0.8        |

### Phase 1: Core Implementation

| Task    | Description                                                          | Effort | Dependency | Status                               |
| ------- | -------------------------------------------------------------------- | ------ | ---------- | ------------------------------------ |
| **1.1** | Implement data module (`ConiferDataset`, `transforms`)               | 2 days | 0.1-0.5    | ✅ DONE                              |
| **1.2** | Implement model factory (`create_model`)                             | 1 day  | 1.1        | ✅ DONE                              |
| **1.3** | Implement training loop with **unfrozen backbone + differential LR** | 2 days | 1.2        | ✅ DONE                              |
| **1.4** | Implement evaluation metrics                                         | 1 day  | 1.1        | ✅ DONE                              |
| **1.5** | Test pipeline with one model (ResNet-50)                             | 1 day  | 1.3, 1.4   | ✅ DONE (retrained on balanced data) |
| **1.6** | Add checkpointing and logging                                        | 1 day  | 1.5        | ✅ DONE                              |

### Phase 2: Model Training (Tuần 3-4)

> **Note**: Fine-tuning (unfrozen backbone) is the default mode. Use `--linear_probe` to fall back to frozen backbone (linear probing).
> **Dataset**: Training now uses `data/task-A/` (7,264 samples, balanced via augmentation) instead of `data/split/` (4,487 samples, imbalanced).
> **Label Smoothing**: All models trained with `label_smoothing=0.1` (default config per design doc).

| Task    | Description                                                         | Status                 |
| ------- | ------------------------------------------------------------------- | ---------------------- |
| **2.1** | Train ResNet-50 (Task A, seed=42, fine-tuning, balanced data)       | ✅ Done                |
| **2.2** | Train EfficientNet-B3 (Task A, seed=42, fine-tuning, balanced data) | ✅ Done                |
| **2.3** | Train MobileNetV3-Large (Task A, seed=42, fine-tuning)              | ✅ Done                |
| **2.4** | Train ConvNeXt-Tiny (Task A, seed=42, fine-tuning)                  | ✅ Done                |
| **2.5** | Train Swin-T (Task A, seed=42, fine-tuning)                         | ✅ Done                |
| **2.6** | Multi-seed: seed=123 (all 5 models)                                 | ✅ Done                |
| **2.7** | Multi-seed: seed=456 (all 5 models)                                 | ✅ Done                |
| **2.8** | Generate results tables and confusion matrices                      | ✅ Done (in notebooks) |

### Phase 3: Efficiency & Ablation (Tuần 5-6)

| Task    | Description                                        | Status                          |
| ------- | -------------------------------------------------- | ------------------------------- |
| **3.1** | Measure inference time for all 5 models (100 runs) | ⚠️ Partial (ConvNeXt ~4ms done) |
| **3.2** | Calculate FLOPs for all 5 models (ptflops)         | ❌ Pending                      |
| **3.3** | Measure model size (MB)                            | ❌ Pending                      |
| **3.4** | Create efficiency comparison table                 | ❌ Pending                      |
| **3.5** | Implement 90° rotation augmentation (code ready)   | ✅ Done                         |
| **3.6** | Run ablation: with vs without 90° rotation         | ❌ Pending (Task B)             |
| **3.7** | Ablation: label smoothing 0.0 vs 0.1 vs 0.2        | ❌ Pending                      |

### Phase 4: Visualization (Tuần 6-7)

| Task    | Description                           | Status                          |
| ------- | ------------------------------------- | ------------------------------- |
| **4.1** | Generate confusion matrix plots (4×4) | ✅ Done (in notebooks)          |
| **4.2** | Generate per-class F1 bar chart       | ✅ Done (in notebooks)          |
| **4.3** | Generate Grad-CAM visualizations      | ⚠️ Code ready, generate pending |
| **4.4** | Create efficiency radar chart         | ❌ Pending                      |

### Task B: Species Classification (20-class)

| Task    | Description                                              | Status |
| ------- | -------------------------------------------------------- | ------ |
| **B.1** | Create data split notebook (stratified by species)       | ✅ Done |
| **B.2** | Create training notebooks for Task B (reuse Task A code) | ❌ Pending |
| **B.3** | Train 5 models × 3 seeds for Task B                      | ❌ Pending |
| **B.4** | Efficiency profiling for Task B models                    | ❌ Pending |
| **B.5** | Ablation: 90° rotation (best model Task B)              | ❌ Pending |
| **B.6** | Ablation: label smoothing 0.0 vs 0.1 vs 0.2             | ❌ Pending |

---

## Dependencies

### Critical Path

```
Phase -1 (Dataset Exploration) ← NEW
    ↓
Phase 0 (Dataset Prep)
    ↓
Phase 1 (Code Implementation)
    ↓
Phase 2 (Training) ─────────────────┐
    ↓                              ↓
Phase 3 (Efficiency)          Phase 3 (Ablation)
    ↓                              ↓
Phase 4 (Multi-seed) ←────────────────┘
    ↓
Phase 5 (Documentation)
```

### Blocking Dependencies

|| Task | Blocked By |
||------|-----------|
|| Phase 0 | Phase -1 (must complete first) |
|| Phase 1 | Phase 0 |
|| Phase 2 | Phase 1 |
|| Phase 3 ablation | Phase 2 (need best model first) |
|| Phase 4 | Phase 2 |
|| Phase 5 | Phase 4 |

### External Dependencies

|| Dependency | Owner | Status |
||------------|-------|--------|
|| IC4SD-Wood-Conifer20v1 Dataset | IC4SD Lab | Delivered |
|| GPU Resources | PTIT Lab / Colab | Available (Colab T4) |
|| Supervisor review (sp. labels) | PGS.TS. Nguyễn Trọng Khánh | Pending |

---

## Timeline & Estimates

> **Current status**: May 21, 2026. Phase -1, 0, 1, 2 (Task A training) are complete. Efficiency profiling and ablation pending.

### Week-by-Week Schedule

|| Week | Phase | Tasks | Deliverables |
||------|-------|-------|--------------|
|| Week 1-2 | Phase 0 | 0.1-0.9 | Dataset ready, split created, environment setup |
|| Week 3-4 | Phase 1 | 1.1-1.6 | Core code, training pipeline |
|| Week 5-6 | Phase 2 | 2.1-2.8 | Task A: All 5 models × 3 seeds trained |
|| Week 7 | Phase 2 | Task B | Task B: All 5 models trained (20-class species) |
|| Week 8 | Phase 3 | 3.1-3.4 | Efficiency: inference time, FLOPs, model size |
|| Week 9 | Phase 3 | 3.5-3.7 | Ablation: 90 degree rotation, label smoothing |
|| Week 10-11 | Phase 4 | 4.1-4.4 | Visualizations, Grad-CAM, radar chart |
|| Week 12 | Phase 5 | 5.1-5.5 | Methods section, results analysis |
|| Week 13-14 | Phase 5 | 5.6 | Paper writing, supervisor review |
|| Week 15-16 | Phase 5 | 5.7 | Final revisions, format, submit |

### Training Time Estimates (T4/Colab)

Fine-tuning with data augmentation, 50 epochs, batch=64. Colab T4 throttles after ~2h.

|| Task | Model | Est. Time (T4) | Note |
||------|-------|----------------|------|
|| Task A | ResNet-50 | ~5-7h | Fastest to converge |
|| Task A | EfficientNet-B3 | ~8-10h | Larger model, slower |
|| Task A | MobileNetV3-L | ~3-4h | Lightweight |
|| Task A | ConvNeXt-Tiny | ~6-8h | ConvNeXt overhead |
|| Task A | Swin-T | ~7-9h | Transformer overhead |
|| Task B | All 5 models | 2-3x Task A | 20 classes, harder task |

**Total GPU-hours (seed=42)**: ~250-300h. **All 3 seeds**: ~750-900h.

**Recommendation**: Use Colab Pro (~100h/month) or local GPU to meet timeline.

---

## Risks & Mitigation

### Technical Risks

|| Risk | Likelihood | Impact | Mitigation |
||------|------------|--------|------------|
|| GPU memory insufficient for batch=64 | Medium | High | Reduce batch to 32, enable gradient checkpointing |
|| Training divergence (loss NaN) | Low | High | Use AMP, warmup LR, gradient clipping |
|| Overfitting on small families | Medium | Medium | Strong augmentation, early stopping |
|| Sp. label ambiguity affects accuracy | Medium | Medium | Report both with/without sp. samples |
|| 90° rotation hurts accuracy | Low | Low | Ablation study will show; if worse, remove |

### Resource Risks

|| Risk | Likelihood | Impact | Mitigation |
||------|------------|--------|------------|
|| GPU unavailable/busy | Medium | High | Use cloud GPU (Colab Pro, Lambda Labs) |
|| Training too slow (>4h/model) | Medium | High | Colab T4 throttles after ~2h; use Colab Pro or local GPU; reduce epochs if needed |
|| Disk space for checkpoints | Low | Low | Store only best + final checkpoint |

### Schedule Risks

|| Risk | Likelihood | Impact | Mitigation |
||------|------------|--------|------------|
|| Supervisor delayed feedback | Medium | Medium | Parallel tasks while waiting |
|| Unexpected bugs | High | Medium | Buffer 1 week in schedule |
|| Phase 0 delay affects all phases | Medium | High | Prioritize Phase 0 completion |

---

## Resources Needed

### Team Members & Roles

|| Role | Name | Responsibilities |
||------|------|------------------|
|| Team Lead | (TBD) | Overall coordination, documentation |
|| ML Engineer 1 | (TBD) | Training pipeline, experiments |
|| ML Engineer 2 | (TBD) | Evaluation, visualization |
|| Supervisor | PGS.TS. Nguyễn Trọng Khánh | Technical guidance, review |

### Hardware Resources

|| Resource | Specification | Duration |
||----------|---------------|----------|
|| GPU | NVIDIA RTX 3080+ or A100 | ~200 hours total |
|| Storage | 50GB+ for checkpoints | Project duration |
|| CPU | For data preprocessing | ~20 hours |

### Software/Tools

|| Tool | Purpose |
||------|---------|
|| Python 3.10+ | Runtime |
|| PyTorch 2.x | Deep learning framework |
|| timm | Model zoo |
|| wandb or tensorboard | Experiment tracking |
|| Git | Version control |
|| Overleaf/LaTeX | Paper writing |

### Documentation

|| Document | Location |
||----------|---------|
|| Requirements | `docs/ai/requirements/conifer-family-classification.md` |
|| Design | `docs/ai/design/conifer-family-classification.md` |
|| Planning | `docs/ai/planning/conifer-family-classification.md` |
|| Implementation | `docs/ai/implementation/conifer-family-classification.md` |
|| Testing | `docs/ai/testing/conifer-family-classification.md` |
|| Results | `experiments/taskA/results/` |

---

## Success Metrics

|| Metric | Target | Actual | Status |
||--------|--------|--------|--------|
|| Phase 0 & -1 completed | Yes | Yes | ✅ Complete |
|| Phase 1 code pipeline | Yes | Code complete | ✅ Complete |
|| All 5 models trained (Task A) | 5/5 | 5/5 × 3 seeds | ✅ Complete |
|| Accuracy >97% | >97% | 98.46%-99.90% | ✅ Complete |
|| F1-macro >96% | >96% | 96.27%-99.70% | ✅ Complete |
|| Multi-seed std <1% | <1% | 0.06%-0.71% | ✅ Complete |
|| Efficiency profiling | Yes | Partial | ⚠️ Pending |
|| Ablation documented | Yes | Pending | ❌ Pending |
|| Grad-CAM generated | Yes | Code ready | ⚠️ Pending |
|| Paper submitted | Yes | - | ❌ Pending |

---

## Notes

### On Seed

A **seed** is a random number used to initialize:

- Weight initialization (if any trainable layers)
- Data shuffling
- Dropout (if used)
- Random augmentation (e.g., which rotation angle)

Using multiple seeds (seed=42, 123, 456) ensures:

1. Results are not dependent on one random initialization
2. Standard deviation captures model variance
3. More robust conclusions

### On Phase 0 Sp. Label Decision

**Strategy**: Keep as-is based on exploration findings. Report results with/without sp. samples in final analysis.

- **Calocedrus sp.** → likely Cupressaceae family (verify in exploration)
- **Pinus sp.** → Pinaceae family (verify in exploration)

Options:

1. Keep as-is (treat as separate species, but parent family is clear) — **Recommended**
2. Merge into known species (if species can be inferred)
3. Exclude from dataset for cleaner experiment

### On Training Strategy: Fine-tuning vs. Linear Probing

Per WoodID_plan.md, **fine-tuning** (backbone unfrozen) with differential learning rates is the **default** mode. The two modes:

|| Mode | Backbone | Head LR | Backbone LR | Trainable Params | Use Case |
||------|----------|---------|-------------|-----------------|----------|
|| **Fine-tuning (default)** | Unfrozen | 1e-4 | 1e-5 | All (~25M) | **Recommended** — better accuracy |
|| Linear probing | Frozen | 1e-4 | — | ~2K (head only) | Fast baseline, ablation reference |

The lower backbone LR (1/10th of head LR) prevents catastrophic forgetting of pretrained ImageNet features while still allowing adaptation to the wood cross-section domain.

CLI usage:

```bash
# Fine-tuning (default — no flag needed)
python scripts/train_taskA.py --backbone resnet50 --epochs 50

# With custom backbone LR
python scripts/train_taskA.py --backbone resnet50 --epochs 50 --backbone_lr 5e-6

# Linear probing (frozen backbone)
python scripts/train_taskA.py --backbone resnet50 --epochs 50 --linear_probe
```

### On Label Smoothing Ablation

Label smoothing = 0.1 is the **default training config** applied to all Task A experiments. The ablation study (0.0 vs 0.1 vs 0.2) is a **separate ablation study** to determine optimal label smoothing value, not part of the main Task A training.
