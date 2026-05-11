---
phase: planning
title: Conifer Family Classification (Task A) — Project Planning
description: Task breakdown and timeline for family-level classification of 4 conifer families
---

# Project Planning & Task Breakdown — Task A: Conifer Family Classification

## Milestones

- [x] **M0: Dataset Exploration** — Understand raw data structure, species list, image quality
- [x] **M1: Phase 0 Complete** — Dataset organized, labels resolved, split created, notebook ready
- [x] **M2: Baseline Pipeline** — Code structure ready, one model trains successfully ✅ *Phase 1 Complete* (model head fixed May 2026)
- [ ] **M3: All 5 Models Trained** — Task A complete for all architectures (fine-tuning, differential LR)
- [ ] **M4: Efficiency Analysis** — Inference time, FLOPs, model size measured
- [ ] **M5: Ablation Complete** — 90° rotation augmentation tested
- [ ] **M6: Multi-seed Validation** — 3 seeds run, mean ± std reported
- [ ] **M7: Visualization Complete** — Confusion matrix, Grad-CAM, per-class F1

---

## Task Breakdown

### Phase -1: Dataset Exploration (NEW — Before Phase 0)

> **Purpose**: Understand raw data before organizing. Not owner of dataset — need to discover what's actually there.

| Task | Description | Effort | Output |
|------|-------------|--------|--------|
| **-1.1** | Explore raw data structure | List folders, count images per folder | 2h | Structure documented |
| **-1.2** | Verify species list | Compare against tree.txt references | 2h | Species list verified |
| **-1.3** | Check image quality | Resolution, format, corruption check | 2h | Quality report |
| **-1.4** | Identify ambiguous labels | Calocedrus sp., Pinus sp., duplicates | 2h | Ambiguity report |
| **-1.5** | Document findings | Create exploration notebook | 4h | `notebooks/dataset_exploration.ipynb` |

**Notebook Structure** (fixed template for scalability):

```markdown
# Dataset Exploration — IC4SD-Wood-Conifer20v1

## 1. Metadata
- Dataset path, total images
- Folder structure
- File formats

## 2. Species Distribution
- Per-species image count
- Per-family image count
- Class imbalance analysis

## 3. Image Analysis
- Resolution distribution
- Sample images per species
- Quality issues (blurry, corrupt, etc.)

## 4. Ambiguous Labels
- Calocedrus sp.
- Pinus sp.
- Duplicates or near-duplicates

## 5. Conclusions & Recommendations
- Species list for Phase 0
- Known issues to handle
- Split strategy suggestions
```

### Phase 0: Preparation (Tuần 1-2)

| Task | Description | Effort | Dependency |
|------|-------------|--------|------------|
| **0.1** | Organize dataset folder structure `data/{split}/{family}/{species}/` | 1 day | -1.x |
| **0.2** | Create `species_to_family.json` label mapping | 1 day | 0.1 |
| **0.3** | Verify species-to-family mapping against exploration findings | 1 day | 0.2 |
| **0.4** | Create stratified split 70/15/15, seed=42 | 1 day | 0.3 |
| **0.5** | Verify split: check class distribution | 1 day | 0.4 |
| **0.6** | Visualize sample images per species (from exploration findings) | 1 day | -1.x |
| **0.7** | Setup Python environment (PyTorch, timm, etc.) | 1 day | None |
| **0.8** | Test GPU availability and CUDA setup | 1 day | 0.7 |
| **0.9** | Profile baseline inference time (dummy model) | 1 day | 0.8 |

**Phase 0 Notebook Deliverable**: Full documentation with fixed structure, all steps with outputs, ready for reproducibility.

### Phase 1: Core Implementation (Tuần 2-3) ✅ DONE

| Task | Description | Effort | Dependency |
|------|-------------|--------|------------|
| **1.1** | Implement data module (`ConiferDataset`, `transforms`) | 2 days | 0.1-0.5 | ✅ |
| **1.2** | Implement model factory (`create_model`) | 1 day | 1.1 | ✅ |
| **1.3** | Implement training loop with **unfrozen backbone + differential LR** | 2 days | 1.2 | ✅ |
| **1.4** | Implement evaluation metrics | 1 day | 1.1 | ✅ |
| **1.5** | Test pipeline with one model (ResNet-50) | 1 day | 1.3, 1.4 | ⏳ |
| **1.6** | Add checkpointing and logging | 1 day | 1.5 | ✅ |

> **Training Strategy**: Fine-tuning with unfrozen backbone + differential learning rates (backbone LR = 1e-5, head LR = 1e-4) is the **default mode** per WoodID_plan.md. This adapts pretrained ImageNet features to the wood cross-section domain while preserving low-level texture features. Use `--linear_probe` flag to fall back to frozen backbone (linear probing) for ablation comparison.

### Phase 2: Model Training (Tuần 3-4)

> **Note**: Fine-tuning (unfrozen backbone) is the default mode. Use `--linear_probe` to fall back to frozen backbone (linear probing).

| Task | Description | Effort | Dependency |
|------|-------------|--------|------------|
| **2.1** | Train ResNet-50 (Task A, seed=42, fine-tuning) | 1 day | 1.6 | ⏳ |
| **2.2** | Train EfficientNet-B3 (Task A, seed=42, fine-tuning) | 1 day | 1.6 | ⏳ |
| **2.3** | Train MobileNetV3-Large (Task A, seed=42, fine-tuning) | 1 day | 1.6 | ⏳ |
| **2.4** | Train ConvNeXt-Tiny (Task A, seed=42, fine-tuning) | 1 day | 1.6 | ⏳ |
| **2.5** | Train Swin-T (Task A, seed=42, fine-tuning) | 1 day | 1.6 | ⏳ |
| **2.6** | Generate results tables and confusion matrices | 1 day | 2.1-2.5 | ⏳ |

### Phase 3: Efficiency & Ablation (Tuần 5-6)

| Task | Description | Effort | Dependency |
|------|-------------|--------|------------|
| **3.1** | Measure inference time for all 5 models (100 runs) | 1 day | 2.1-2.5 |
| **3.2** | Calculate FLOPs for all 5 models (ptflops) | 1 day | 2.1-2.5 |
| **3.3** | Measure model size (MB) | 1 day | 2.1-2.5 |
| **3.4** | Create efficiency comparison table | 1 day | 3.1-3.3 |
| **3.5** | Implement 90° rotation augmentation | 1 day | 2.6 |
| **3.6** | Run ablation: with vs without 90° rotation | 2 days | 3.5, best model |

### Phase 4: Multi-seed & Visualization (Tuần 6-7)

| Task | Description | Effort | Dependency |
|------|-------------|--------|------------|
| **4.1** | Run all 5 models with seed=123 | 3 days | 2.1-2.5 |
| **4.2** | Run all 5 models with seed=456 | 3 days | 2.1-2.5 |
| **4.3** | Calculate mean ± std for all metrics | 1 day | 4.1, 4.2 |
| **4.4** | Generate confusion matrix plots (4×4) | 1 day | 4.3 |
| **4.5** | Generate per-class F1 bar chart | 1 day | 4.3 |
| **4.6** | Generate Grad-CAM visualizations | 1 day | 4.3 |
| **4.7** | Create efficiency radar chart | 1 day | 3.4 |

### Phase 5: Documentation & Analysis (Tuần 7-8)

| Task | Description | Effort | Dependency |
|------|-------------|--------|------------|
| **5.1** | Write Section 3 (Methods) for Task A results | 2 days | 4.4-4.7 |
| **5.2** | Analyze inter-family confusion patterns | 1 day | 4.4 |
| **5.3** | Analyze efficiency-accuracy trade-off | 1 day | 3.4 |
| **5.4** | Prepare figures for paper (300 DPI, captions) | 1 day | 4.4-4.7 |
| **5.5** | Final results summary table | 1 day | 4.3 |

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

| Task | Blocked By |
|------|-----------|
| Phase 0 | Phase -1 (must complete first) |
| Phase 1 | Phase 0 |
| Phase 2 | Phase 1 |
| Phase 3 ablation | Phase 2 (need best model first) |
| Phase 4 | Phase 2 |
| Phase 5 | Phase 4 |

### External Dependencies

| Dependency | Owner | Status |
|------------|-------|--------|
| IC4SD-Wood-Conifer20v1 Dataset | IC4SD Lab | Delivered |
| GPU Resources | PTIT Lab / Colab | Available (Colab T4) |
| Supervisor review (sp. labels) | PGS.TS. Nguyễn Trọng Khánh | Pending |

---

## Timeline & Estimates

> **Current status**: May 11, 2026. Phase -1, 0, 1 (core implementation) are complete. Phase 2 (training) is in progress. Remaining ~12-16 weeks until target submit 9-10/2026.

### Week-by-Week Schedule

| Week | Phase | Tasks | Deliverables |
|------|-------|-------|--------------|
| Week 1-2 | Phase 0 | 0.1-0.9 | Dataset ready, split created, environment setup |
| Week 3-4 | Phase 1 | 1.1-1.6 | Core code, training pipeline |
| Week 5-6 | Phase 2 | 2.1-2.6 | Task A: All 5 models trained, results tables |
| Week 7 | Phase 2 | 2.7-2.9 | Task B: All 5 models trained (20-class species) |
| Week 8 | Phase 3 | 3.1-3.4 | Efficiency: inference time, FLOPs, model size |
| Week 9 | Phase 3 | 3.5-3.6 | Ablation: 90 degree rotation augmentation |
| Week 10-11 | Phase 4 | 4.1-4.7 | Multi-seed (3 seeds), visualizations, Grad-CAM |
| Week 12 | Phase 5 | 5.1-5.5 | Methods section, results analysis |
| Week 13-14 | Phase 5 | 5.6 | Paper writing, supervisor review |
| Week 15-16 | Phase 5 | 5.7 | Final revisions, format, submit |

### Training Time Estimates (T4/Colab)

Fine-tuning with data augmentation, 50 epochs, batch=64. Colab T4 throttles after ~2h.

| Task | Model | Est. Time (T4) | Note |
|------|-------|----------------|------|
| Task A | ResNet-50 | ~5-7h | Fastest to converge |
| Task A | EfficientNet-B3 | ~8-10h | Larger model, slower |
| Task A | MobileNetV3-L | ~3-4h | Lightweight |
| Task A | ConvNeXt-Tiny | ~6-8h | ConvNeXt overhead |
| Task A | Swin-T | ~7-9h | Transformer overhead |
| Task B | All 5 models | 2-3x Task A | 20 classes, harder task |

**Total GPU-hours (seed=42)**: ~250-300h. **All 3 seeds**: ~750-900h.

**Recommendation**: Use Colab Pro (~100h/month) or local GPU to meet timeline.

---

## Risks & Mitigation

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| GPU memory insufficient for batch=64 | Medium | High | Reduce batch to 32, enable gradient checkpointing |
| Training divergence (loss NaN) | Low | High | Use AMP, warmup LR, gradient clipping |
| Overfitting on small families | Medium | Medium | Strong augmentation, early stopping |
| Sp. label ambiguity affects accuracy | Medium | Medium | Report both with/without sp. samples |
| 90° rotation hurts accuracy | Low | Low | Ablation study will show; if worse, remove |

### Resource Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| GPU unavailable/busy | Medium | High | Use cloud GPU (Colab Pro, Lambda Labs) |
| Training too slow (>4h/model) | Medium | High | Colab T4 throttles after ~2h; use Colab Pro or local GPU; reduce epochs if needed |
| Disk space for checkpoints | Low | Low | Store only best + final checkpoint |

### Schedule Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Supervisor delayed feedback | Medium | Medium | Parallel tasks while waiting |
| Unexpected bugs | High | Medium | Buffer 1 week in schedule |
| Phase 0 delay affects all phases | Medium | High | Prioritize Phase 0 completion |

---

## Resources Needed

### Team Members & Roles

| Role | Name | Responsibilities |
|------|------|------------------|
| Team Lead | (TBD) | Overall coordination, documentation |
| ML Engineer 1 | (TBD) | Training pipeline, experiments |
| ML Engineer 2 | (TBD) | Evaluation, visualization |
| Supervisor | PGS.TS. Nguyễn Trọng Khánh | Technical guidance, review |

### Hardware Resources

| Resource | Specification | Duration |
|----------|---------------|----------|
| GPU | NVIDIA RTX 3080+ or A100 | ~200 hours total |
| Storage | 50GB+ for checkpoints | Project duration |
| CPU | For data preprocessing | ~20 hours |

### Software/Tools

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Runtime |
| PyTorch 2.x | Deep learning framework |
| timm | Model zoo |
| wandb or tensorboard | Experiment tracking |
| Git | Version control |
| Overleaf/LaTeX | Paper writing |

### Documentation

| Document | Location |
|----------|----------|
| Requirements | `docs/ai/requirements/conifer-family-classification.md` |
| Design | `docs/ai/design/conifer-family-classification.md` |
| Planning | `docs/ai/planning/conifer-family-classification.md` |
| Implementation | `docs/ai/implementation/conifer-family-classification.md` |
| Testing | `docs/ai/testing/conifer-family-classification.md` |
| Results | `experiments/taskA/results/` |

---

## Success Metrics

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Phase 0 & -1 completed | Yes | Yes | ✅ Complete |
| Phase 1 code pipeline | Yes | Yes (fixed) | ✅ Complete |
| All 5 models trained (Task A) | 5/5 | - | Pending |
| Accuracy >97% | >97% | - | Pending |
| F1-macro >96% | >96% | - | Pending |
| Multi-seed std <1% | <1% | - | Pending |
| Ablation documented | Yes | - | Pending |
| Grad-CAM generated | Yes | - | Pending |
| Paper submitted | Yes | - | Pending |

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

**Note**: Q4 cannot be answered until Phase -1 (Dataset Exploration) is complete.

### On Training Strategy: Fine-tuning vs. Linear Probing

Per WoodID_plan.md, **fine-tuning** (backbone unfrozen) with differential learning rates is the **default** mode. The two modes:

| Mode | Backbone | Head LR | Backbone LR | Trainable Params | Use Case |
|------|----------|---------|-------------|-----------------|----------|
| **Fine-tuning (default)** | Unfrozen | 1e-4 | 1e-5 | All (~25M) | **Recommended** — better accuracy |
| Linear probing | Frozen | 1e-4 | — | ~2K (head only) | Fast baseline, ablation reference |

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
