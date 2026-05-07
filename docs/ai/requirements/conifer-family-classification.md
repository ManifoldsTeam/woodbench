---
phase: requirements
title: Conifer Family Classification (Task A)
description: Family-level classification of 4 conifer families (Cupressaceae, Pinaceae, Podocarpaceae, Taxaceae) using 20-species dataset
---

# Requirements & Problem Understanding — Task A: Conifer Family Classification

## Problem Statement

**What problem are we solving?**

Conifer wood species identification is critical for Vietnam's timber import inspection at border checkpoints. However, macroscopic cross-section images of conifer wood from different families (Cupressaceae, Pinaceae, Podocarpaceae, Taxaceae) share visual similarities that make manual identification error-prone and time-consuming.

Task A is the **family-level classification subtask** of the WoodID-S2-2026 benchmark project:
- Classify microscopic/macroscopic wood cross-section images into 4 botanical families
- This serves as a pipeline validation step before the more challenging species-level Task B
- Provide efficiency metrics (inference time, model size, FLOPs) for real-world deployment decisions

**Who is affected?**
- Customs officers and border inspection authorities needing rapid timber identification
- Wood industry quality control personnel
- Conservation authorities monitoring CITES-listed species (Taxus wallichiana)

**Current situation/workaround:**
- Manual identification by expert botanists — slow, expensive, not scalable
- No standardized deep learning benchmark for conifer family classification in Vietnam context

---

## Goals & Objectives

### Primary Goals

1. **Benchmark 5 SOTA architectures** on 4-class family classification:
   - ResNet-50, EfficientNet-B3, MobileNetV3-Large, ConvNeXt-Tiny, Swin-T

2. **Achieve high accuracy** (target: >97% accuracy) for family-level classification

3. **Comprehensive efficiency analysis**: Measure and compare inference time (ms/image), model size (MB), FLOPs for deployment suitability

4. **Ablation study on augmentation**: Test discrete 90° rotation augmentation hypothesis — cross-section images are isotropic (no natural orientation), so 0°, 90°, 180°, 270° rotations should preserve information and potentially improve accuracy

### Secondary Goals

5. **Reproducibility**: Run experiments with 3 different seeds, report mean ± std

6. **Explainability**: Generate Grad-CAM visualizations to understand what features each model focuses on

7. **Pipeline validation**: Task A validates the training pipeline before scaling to Task B (20-class)

### Non-goals (out of scope)

- Species-level classification (Task B) — separate feature
- Mobile/edge deployment implementation — just evaluation metrics
- Real-time application UI — just benchmarking
- Dataset collection — using existing IC4SD-Wood-Conifer20v1 dataset

---

## User Stories & Use Cases

### Primary User: Research Team

- **As a researcher**, I want standardized benchmark results so I can compare model architectures fairly
  - *Workflow*: Run 5 models → Get comparable metrics → Publish comparison paper

- **As a researcher**, I want efficiency metrics (speed, size) to recommend models for different deployment scenarios
  - *Workflow*: Evaluate models → Compare efficiency → Recommend for edge vs. cloud deployment

### Secondary User: Customs/Border Inspection (Future Application)

- **As a customs officer**, I want fast, accurate family-level classification to quickly categorize timber shipments
  - *Workflow*: Capture/upload image → Get family prediction → Flag suspicious shipments for expert review

- **As a customs officer**, I want confidence scores to know when to trust the model's prediction
  - *Workflow*: View prediction → See confidence bar → Decide to accept or escalate

### Edge Cases

- Image quality issues: blurry, too dark/bright, partial wood cross-section
- Ambiguous samples near family boundaries
- Novel species not in training set (Cupressaceae vs Pinaceae confusion)

---

## Success Criteria

### Phase 0 Deliverables (Pre-Experiment)

- [ ] Dataset folder structure: `data/{train,val,test}/{family}/{species}/`
- [ ] `species_to_family.json` label mapping created
- [ ] Stratified split 70/15/15, seed=42, verified
- [ ] Calocedrus sp. và Pinus sp. label assignment resolved (keep as-is)

### Measurable Outcomes

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Top-1 Accuracy | >97% | Eval on test set (seed=42) |
| F1-macro | >96% | sklearn.metrics.f1_score |
| Precision (macro) | >96% | sklearn.metrics.precision_score |
| Recall (macro) | >96% | sklearn.metrics.recall_score |
| Inference Time | <50ms/image | 100 runs, batch=1, GPU |
| Model Size | <100MB (for lightweight models) | File size of .pth |

### Acceptance Criteria

- [ ] All 5 models trained successfully for Task A (4-class family)
- [ ] Confusion matrix shows minimal inter-family confusion (expected: Podocarpaceae/Taxaceae vs others)
- [ ] Per-class F1 reported for all 4 families
- [ ] Efficiency comparison table: Params / FLOPs / Inference Time / Size / VRAM
- [ ] Ablation study: 90° rotation on/off comparison
- [ ] 3 seeds run → mean ± std reported
- [ ] Grad-CAM visualizations for each family

### Performance Benchmarks

- Training: ~2-4 hours per model on RTX 3080+ or equivalent
- Total experiments: 5 models × 2 augmentation configs × 3 seeds = 30 training runs
- Evaluation: <5 minutes per model on test set

---

## Constraints & Assumptions

### Technical Constraints

- Dataset: IC4SD-Wood-Conifer20v1 — 7,050 images, 20 species, 4 families
  - Split: 70/15/15 (train/val/test), stratified by species, seed=42
  - ~4,935 train | ~1,057 val | ~1,058 test images
  - **Phase 0 Deliverable**: Create `species_to_family.json` label mapping
- Image size: 224×224 (standard for pretrained backbones)
- 2 species with uncertain labels: Calocedrus sp., Pinus sp. → Keep as-is (Phase 0 decision)
- GPU required: NVIDIA GPU with CUDA 12.x, >=8GB VRAM recommended

### Training Configuration Constraints

- Backbone: ImageNet pretrained, frozen (only head trained)
- Optimizer: AdamW, lr=1e-4
- Scheduler: CosineAnnealingLR, T_max=50
- Epochs: 50
- Batch size: 64
- Mixed precision (AMP): enabled
- Label smoothing: 0.1

### Business/Research Constraints

- Timeline: Phase 1 (Tuần 3-4) — Task A experiments
- Paper target: Q2 journal (Signal, Image and Video Processing or IET Image Processing)
- No external data collection — must use existing IC4SD dataset only

### Assumptions

- ImageNet pretrained weights provide sufficient transfer for macroscopic wood textures
- Family-level classification is easier than species-level (4 vs 20 classes) — should achieve >97%
- Cross-section images are isotropic — 90° rotations preserve semantic information
- 7 loài endemic VN và 1 CITES species không affect family-level difficulty significantly

---

## Reference Documents (Tài liệu gốc)

**RULE**: Các tài liệu sau là tài liệu gốc và **PHẢI đối chiếu** sau khi có kết quả experiment:

| Document | Purpose |
|---------|---------|
| `docs/references/WoodID_plan.md` | Project plan, experiment checklist, timeline |
| `docs/references/WoodID_Assignment_S2.docx` | Research assignment specification |
| `docs/references/tree.txt` | IC4SD research assignment document |

---

## Questions & Open Items

### Resolved Questions

1. **Calocedrus sp. và Pinus sp.**: Keep as-is (Calocedrus sp. → Cupressaceae, Pinus sp. → Pinaceae) — follows Phase 0 decision

2. **90° Rotation Augmentation**: Discrete rotations — randomly select one of 4 orientations (0°, 90°, 180°, 270°) with equal probability

3. **Label Mapping JSON**: Create Phase 0 deliverable `species_to_family.json` with format:
   ```json
   {
     "cunninghamia_lanceolata": "Cupressaceae",
     "callitris_columellaris": "Cupressaceae",
     ...
     "taxus_wallichiana": "Taxaceae"
   }
   ```

### Unresolved Questions

4. **Grad-CAM Layer Selection**: Which layer should be used as target for Grad-CAM?
   - Final convolutional layer of each backbone (per design doc)
   - *Status*: Will define per-model in implementation

5. **Best Model Selection**: Which metric determines "best model"?
   - Report all metrics, recommend based on use case
   - Accuracy-first vs efficiency-first scenarios

### Items Requiring Stakeholder Input

- [ ] Supervisor review of family-level accuracy targets (>97%)
- [ ] Decision on whether to include t-SNE visualization for Task A

---

## Reference Documents Verification

**RULE**: Sau khi hoàn thành experiment, **PHẢI đối chiếu** kết quả với tài liệu gốc:

| Document | Checkpoints |
|----------|-------------|
| `docs/references/WoodID_plan.md` | Experiment checklist, Phase 1 metrics alignment |
| `docs/references/WoodID_Assignment_S2.docx` | Research requirements compliance |
| `docs/references/tree.txt` | IC4SD assignment requirements |

---

## References

- WoodID_plan.md — Full project plan (PRIMARY REFERENCE)
- WoodID_Assignment_S2.docx — Research assignment specification (PRIMARY REFERENCE)
- tree.txt — IC4SD research assignment document (PRIMARY REFERENCE)
- Related work: Ravindran et al. (2020), Lens et al. (2020), Filho et al. (2022)
