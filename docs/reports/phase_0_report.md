# WoodID-S2-2026 — Phase 0 Report & GPU Cost Analysis

**Project**: A Systematic Benchmark of Deep Learning Architectures for Conifer Wood Species Identification  
**Task**: Task A — Family-Level Classification  
**Date**: May 7, 2026  
**Status**: Phase 0 Complete ✅

---

## Phần 1: Phase 0 Report — Task A

### 1.1 Tổng Quan Phase 0

Phase 0 (Chuẩn bị) đã hoàn thành theo đúng checklist trong `WoodID_plan.md`:

| #   | Nhiệm vụ                                                    | Trạng thái           |
| --- | ----------------------------------------------------------- | -------------------- |
| 1   | Tổ chức folder 2 cấp: `data/{split}/{family}/{species}/`    | ✅ Done              |
| 2   | Tạo label mapping JSON: `species_name → family_name`        | ✅ Done              |
| 3   | Quyết định xử lý Calocedrus sp. và Pinus sp.                | ✅ Done (keep as-is) |
| 4   | Stratified split 70/15/15 theo species, seed=42             | ✅ Done              |
| 5   | Visualize 4 ảnh mỗi loài, sắp xếp theo family               | ✅ Done              |
| 6   | Setup môi trường, test GPU, profile inference time baseline | ✅ Done              |

### 1.2 Dataset Overview — IC4SD-Wood-Conifer20v1

#### Dataset Statistics (Thực tế vs. Ước tính)

| Metric             | Design (WoodID_plan.md) | Actual    | Difference   |
| ------------------ | ----------------------- | --------- | ------------ |
| Total images       | 7,050                   | **6,423** | -627 (-8.9%) |
| Number of species  | 20                      | 20        | ✅           |
| Number of families | 4                       | 4         | ✅           |
| Image dimensions   | 50× macro (raw)         | 2560×2048 | ✅           |
| Corrupted images   | 0                       | 0         | ✅           |

**Note**: Dataset thực tế nhỏ hơn ước tính ~9%. Điều này không ảnh hưởng đến chất lượng experiment vì:

- Tất cả 20 loài đều có mặt
- Phân bố family vẫn cân đối
- Stratified split vẫn đảm bảo tỷ lệ

#### Per-Species Image Count

| #   | Family        | Species                       | Raw Images |
| --- | ------------- | ----------------------------- | ---------- |
| 1   | Cupressaceae  | Cunninghamia lanceolata       | 450        |
| 2   | Cupressaceae  | Callitris columellaris        | 416        |
| 3   | Cupressaceae  | Fokienia hodginsii ⭐         | 412        |
| 4   | Cupressaceae  | Cupressus funebris            | 318        |
| 5   | Cupressaceae  | Calocedrus sp. ⚠️             | 238        |
| 6   | Cupressaceae  | Cryptomeria japonica          | 335        |
| 7   | Cupressaceae  | Xanthocyparis vietnamensis ⭐ | 318        |
| 8   | Cupressaceae  | Chamaecyparis obtusa          | 317        |
| 9   | Cupressaceae  | Calocedrus rupestris ⭐       | 302        |
| 10  | Cupressaceae  | Thujopsis dolabrata           | 327        |
| 11  | Pinaceae      | Pinus dalatensis ⭐           | 334        |
| 12  | Pinaceae      | Pinus radiata                 | 332        |
| 13  | Pinaceae      | Pinus sp. ⚠️                  | 216        |
| 14  | Pinaceae      | Pinus krempfii ⭐             | 323        |
| 15  | Pinaceae      | Picea glauca                  | 304        |
| 16  | Pinaceae      | Araucaria araucana            | 300        |
| 17  | Pinaceae      | Pseudotsuga brevifolia ⭐     | 289        |
| 18  | Pinaceae      | Pinus cernua ⭐               | 287        |
| 19  | Podocarpaceae | Podocarpus pilgeri            | 310        |
| 20  | Taxaceae      | Taxus wallichiana ⭐ (CITES)  | 295        |

⭐ = Endemic Vietnam / CITES species  
⚠️ = Species-level ambiguous

### 1.3 Data Split Distribution

**Split Configuration**:

- Method: Stratified (theo species)
- Ratios: 70% / 15% / 15% (train / val / test)
- Seed: 42

#### Split Summary

| Split     | Count | Percentage |
| --------- | ----- | ---------- |
| **Train** | 4,487 | 69.9%      |
| **Val**   | 955   | 14.9%      |
| **Test**  | 981   | 15.3%      |
| **Total** | 6,423 | 100%       |

#### Per-Family Distribution

| Family        | Train | Val | Test | Total | % Total |
| ------------- | ----- | --- | ---- | ----- | ------- |
| Cupressaceae  | 2,398 | 510 | 525  | 3,433 | 53.5%   |
| Pinaceae      | 1,666 | 355 | 364  | 2,385 | 37.1%   |
| Podocarpaceae | 217   | 46  | 47   | 310   | 4.8%    |
| Taxaceae      | 206   | 44  | 45   | 295   | 4.6%    |

### 1.4 Label Mapping

File: `data/species_to_family.json`

```json
{
  "families": {
    "Cupressaceae": 0,
    "Pinaceae": 1,
    "Podocarpaceae": 2,
    "Taxaceae": 3
  },
  "mapping": {
    "3244.Calocedrus sp": 0,
    "3252.Callitris columellaris": 0,
    ...
  }
}
```

**Key Decisions**:

- ✅ Calocedrus sp. → Cupressaceae (keep as-is)
- ✅ Pinus sp. → Pinaceae (keep as-is)

### 1.5 Quality Assurance Results

| Check             | Result                   |
| ----------------- | ------------------------ |
| Dataset structure | ✅ All folders present   |
| Image counts      | ✅ 6,423/6,423 (100%)    |
| Corrupted images  | ✅ 0 found (sampled 101) |
| Broken symlinks   | ✅ 0 found               |
| Symlink integrity | ✅ 6,423 valid           |

### 1.6 Phase 0 Outputs

| File/Directory                        | Description    | Status |
| ------------------------------------- | -------------- | ------ |
| `data/split/train/`                   | Training set   | ✅     |
| `data/split/val/`                     | Validation set | ✅     |
| `data/split/test/`                    | Test set       | ✅     |
| `data/species_to_family.json`         | Label mapping  | ✅     |
| `data/split/split_metadata.json`      | Split metadata | ✅     |
| `data/split/dataset_manifest.csv`     | Image manifest | ✅     |
| `data/split/qa_summary.json`          | QA report      | ✅     |
| `data/split/split_distribution.png`   | Visualization  | ✅     |
| `data/split/species_distribution.png` | Visualization  | ✅     |

---

## Phần 2: Dataset Research Chi Tiết

### 2.1 Dataset Characteristics

**Image Properties**:

- Resolution: 2560×2048 pixels (fixed)
- Magnification: 50× macro cross-section
- Format: PNG
- Color: Grayscale/RGB (tùy species)

**Domain Characteristics**:

- **Isotropic**: Cross-section images không có "hướng tự nhiên" → VFlip, HFlip, 90° rotation đều hợp lệ
- **Texture-based**: Phân biệt dựa trên wood anatomy (resin canal, tracheid, ray cell)
- **Fine-grained**: Sự khác biệt giữa các loài trong cùng family có thể rất nhỏ

### 2.2 Species Analysis

**Cupressaceae (10 species, 3,433 images)**:

- Nhiều species nhất
- Ưu thế: Cunninghamia lanceolata (450 images)
- Thách thức: Calocedrus sp. (238 images) — label ambiguous
- Novelty: 3 endemic VN species + 1 CITES species

**Pinaceae (8 species, 2,385 images)**:

- Đa dạng: 7 Pinus species + Picea + Araucaria + Pseudotsuga
- Thách thức: Pinus sp. (216 images) — label ambiguous
- 5/8 species là endemic VN

**Podocarpaceae (1 species, 310 images)**:

- Chỉ có Podocarpus pilgeri
- Minority class — có thể ảnh hưởng đến recall

**Taxaceae (1 species, 295 images)**:

- Chỉ có Taxus wallichiana
- CITES Appendix II species
- Minority class — có thể ảnh hưởng đến recall

### 2.3 Class Imbalance Analysis

| Family        | Images | % Total | Imbalance Ratio   |
| ------------- | ------ | ------- | ----------------- |
| Cupressaceae  | 3,433  | 53.5%   | 11.6x vs Taxaceae |
| Pinaceae      | 2,385  | 37.1%   | 8.1x vs Taxaceae  |
| Podocarpaceae | 310    | 4.8%    | 1.05x vs Taxaceae |
| Taxaceae      | 295    | 4.6%    | 1.0x (baseline)   |

**Imbalance ratio: 11.6:1** (Cupressaceae vs Taxaceae)

**Recommendations for Training**:

1. Sử dụng weighted loss hoặc oversampling cho minority classes
2. Monitor per-class F1, không chỉ overall accuracy
3. Confusion matrix sẽ quan trọng để phát hiện bias

---

## Phần 3: GPU Options & Cost Comparison

### 3.1 Machine Specifications (Local)

| Spec | Value                                 |
| ---- | ------------------------------------- |
| CPU  | Intel i5-14600KF (20 cores) @ 5.3 GHz |
| GPU  | NVIDIA RTX 3060 Lite Hash Rate        |
| VRAM | ~12 GB                                |
| RAM  | 31.12 GB                              |
| Disk | 116.60 GB                             |
| OS   | Arch Linux                            |

**RTX 3060 Performance**:

- Training speed: ~70-80% của RTX 3080
- VRAM: Đủ cho batch_size=64 với frozen backbone
- Cost: $0 (local, điện sinh hoạt)

### 3.2 GPU Options Comparison

#### Option 1: Local Machine (RTX 3060)

| Metric         | Value               |
| -------------- | ------------------- |
| VRAM           | ~12 GB              |
| Training speed | Baseline (1.0x)     |
| Cost           | $0                  |
| Time limit     | None                |
| Reliability    | High                |
| Setup required | Yes (CUDA, PyTorch) |

#### Option 2: Kaggle (Free Tier)

| Metric         | Value                     |
| -------------- | ------------------------- |
| GPU            | Tesla T4 16GB             |
| Weekly limit   | 30 hours                  |
| Reset          | Every Saturday 00:00 CET  |
| Training speed | ~1.3x RTX 3060            |
| Cost           | $0                        |
| Time limit     | **30h/week**              |
| Reliability    | Medium (timeout possible) |
| Data transfer  | Upload required           |

#### Option 3: Google Colab Pro

| Metric         | Value                  |
| -------------- | ---------------------- |
| GPU            | T4 or A100 (priority)  |
| Monthly limit  | 500 compute units      |
| Cost           | **$12.67/month**       |
| Training speed | 1.3x-2x RTX 3060       |
| Time limit     | Compute unit dependent |
| Reliability    | High                   |
| Data transfer  | Upload required        |

#### Option 4: Google Colab Pro+

| Metric         | Value                  |
| -------------- | ---------------------- |
| GPU            | A100 priority access   |
| Monthly limit  | 600 compute units      |
| Cost           | **$49.99/month**       |
| Training speed | 3-4x RTX 3060          |
| Time limit     | Compute unit dependent |
| Reliability    | High                   |
| Data transfer  | Upload required        |

#### Option 5: Lambda Labs

| GPU       | VRAM | Cost/hr     | Cost/month (24/7) |
| --------- | ---- | ----------- | ----------------- |
| RTX 6000  | 48GB | $0.50       | $360              |
| A6000     | 48GB | $0.80       | $576              |
| A100 40GB | 40GB | $1.29       | ~$929             |
| A100 80GB | 80GB | $1.79-$2.79 | ~$1,288-$2,006    |

#### Option 6: Vast.ai

| GPU      | VRAM | Cost/hr (On-Demand) | Cost/hr (Interruptible) |
| -------- | ---- | ------------------- | ----------------------- |
| RTX 3090 | 24GB | ~$0.12              | ~$0.08                  |
| RTX 4090 | 24GB | ~$0.28              | ~$0.18                  |
| A100     | 40GB | ~$0.50-$1.00        | ~$0.30-$0.60            |

#### Option 7: RunPod

| GPU       | VRAM | Cost/hr (Community) | Cost/hr (Secure) |
| --------- | ---- | ------------------- | ---------------- |
| A100 40GB | 40GB | $1.19               | $1.39            |
| A100 80GB | 80GB | $1.39               | $1.49-$1.99      |

---

## Phần 4: Training Time Estimation

### 4.1 Experiment Configuration

Based on `WoodID_plan.md` and `conifer-family-classification.md`:

| Parameter       | Value                      |
| --------------- | -------------------------- |
| Epochs          | 50                         |
| Batch size      | 64                         |
| Optimizer       | AdamW (lr=1e-4)            |
| Scheduler       | CosineAnnealingLR          |
| Label smoothing | 0.1                        |
| AMP             | Enabled                    |
| Backbone        | Frozen (transfer learning) |

### 4.2 Model Specifications

| Model           | Parameters | FLOPs (G) | VRAM (MB) | Model Size (MB) |
| --------------- | ---------- | --------- | --------- | --------------- |
| ResNet-50       | 25.6M      | 4.1       | ~3,500    | ~98             |
| EfficientNet-B3 | 12.2M      | 1.8       | ~2,800    | ~47             |
| MobileNetV3-L   | 5.4M       | 0.22      | ~1,800    | ~21             |
| ConvNeXt-T      | 28M        | 4.5       | ~3,800    | ~110            |
| Swin-T          | 28M        | 4.5       | ~4,200    | ~110            |

### 4.3 Training Time Per Model (RTX 3060)

| Model           | Time/Epoch | Task A (50 epochs) | Task B (50 epochs) | 3 Seeds Total |
| --------------- | ---------- | ------------------ | ------------------ | ------------- |
| ResNet-50       | ~90s       | ~75 min            | ~75 min            | ~7.5h         |
| EfficientNet-B3 | ~120s      | ~100 min           | ~100 min           | ~10h          |
| MobileNetV3-L   | ~60s       | ~50 min            | ~50 min            | ~5h           |
| ConvNeXt-T      | ~150s      | ~125 min           | ~125 min           | ~12.5h        |
| Swin-T          | ~180s      | ~150 min           | ~150 min           | ~15h          |

**Task A Total**: ~12.5 hours (5 models, 1 seed each)  
**Task A Total (3 seeds)**: ~37.5 hours  
**Task B Total (3 seeds)**: ~37.5 hours  
**Grand Total (Task A + B, 3 seeds)**: ~75 hours (~3 days)

### 4.4 Training Time by GPU Option

#### Scenario: Full Experiment (Task A + Task B, 3 seeds = 30 runs)

| GPU                  | Time/run | Total Time        | Notes                         |
| -------------------- | -------- | ----------------- | ----------------------------- |
| **RTX 3060 (Local)** | 2.5h     | ~75h (~3 days)    | Free, no time limit           |
| **Kaggle T4**        | ~2h      | ~60h (~2.5 days)  | 30h/week limit → need 2 weeks |
| **Colab Pro T4**     | ~2h      | ~60h (~2.5 days)  | Limited by compute units      |
| **Colab Pro+ A100**  | ~1h      | ~30h (~1.25 days) | ~150 compute units needed     |
| **Vast.ai RTX 3090** | ~1.5h    | ~45h (~1.9 days)  | ~$5.40-$13.50                 |
| **Lambda A100 40GB** | ~1h      | ~30h (~1.25 days) | ~$38.70                       |
| **Lambda RTX 6000**  | ~1.5h    | ~45h (~1.9 days)  | ~$22.50                       |

### 4.5 Cost Analysis for Full Experiment

| Provider             | GPU       | Time | Total Cost                            |
| -------------------- | --------- | ---- | ------------------------------------- |
| **Local (RTX 3060)** | -         | 75h  | **$0** (electricity ~$5-10)           |
| **Kaggle Free**      | T4        | 60h  | **$0** (30h/week × 2 weeks)           |
| **Colab Pro**        | T4        | 60h  | **$12.67**                            |
| **Colab Pro+**       | A100      | 30h  | **$25-50** (depends on compute usage) |
| **Vast.ai**          | RTX 3090  | 45h  | **$5.40-$13.50**                      |
| **Lambda**           | RTX 6000  | 45h  | **$22.50**                            |
| **Lambda**           | A100 40GB | 30h  | **$38.70**                            |
| **RunPod**           | A100 40GB | 30h  | **$35.70**                            |

---

## Phần 5: Recommendations

### 5.1 GPU Selection Matrix

| Priority         | Recommended Option                      | Rationale                           |
| ---------------- | --------------------------------------- | ----------------------------------- |
| **Budget-first** | Local RTX 3060                          | Free, reliable, no limits           |
| **Speed-first**  | Colab Pro+ (A100) or Vast.ai (RTX 3090) | Fastest completion time             |
| **Balance**      | Kaggle Free + Local                     | Combine 30h Kaggle + local training |
| **No-sleep**     | Vast.ai (interruptible)                 | Cheapest for continuous training    |

### 5.2 Recommended Strategy

**Phase 1 Strategy** (for 2-week completion):

```
Week 1:
├── Day 1-2: Task A experiments on local (RTX 3060)
│   └── Train all 5 models (seed=42) → 12.5h
├── Day 3-4: Task B experiments on Kaggle (30h free)
│   └── Train all 5 models (seed=42) → 10h
└── Day 5-7: Remaining seeds on local
    └── Seeds 123, 456 for Task A + B → 50h

Week 2:
├── Day 1-3: Complete remaining training
├── Day 4-5: Ablation studies
└── Day 6-7: Analysis & visualization
```

### 5.3 Cost Optimization Tips

1. **Use Kaggle for initial experiments** (free 30h/week)
2. **Use local machine for overnight training** (RTX 3060, no time limit)
3. **Use Vast.ai interruptible instances** for non-critical training
4. **Save checkpoints frequently** to avoid losing progress on timeouts
5. **Use wandb for experiment tracking** to resume interrupted runs

### 5.4 Technical Recommendations

**For RTX 3060 (Local)**:

```python
# Training config for RTX 3060
batch_size = 64  # Fits in 12GB VRAM
num_workers = 4  # For i5-14600KF
use_amp = True  # Enable for 2x speedup
```

**For Kaggle/Colab (T4)**:

```python
# Training config for T4 (16GB VRAM)
batch_size = 64  # Same, T4 has more VRAM
num_workers = 2  # Cloud CPU limited
use_amp = True  # Essential for speed
```

---

## Phần 6: Appendix

### A. Timeline Summary

| Phase             | Duration | Start        | End          | Status      |
| ----------------- | -------- | ------------ | ------------ | ----------- |
| Phase 0           | 2 weeks  | Apr 2026     | May 7, 2026  | ✅ Complete |
| Phase 1           | 6 weeks  | May 8, 2026  | Jun 19, 2026 | ⏳ Pending  |
| Phase 2           | 4 weeks  | Jun 20, 2026 | Jul 18, 2026 | ⏳ Pending  |
| Phase 3           | 8 weeks  | Jul 19, 2026 | Sep 12, 2026 | ⏳ Pending  |
| **Target Submit** | -        | -            | **Oct 2026** | -           |

### B. Files Generated

```
data/
├── raw/                          # Original dataset
├── split/                        # Phase 0 output
│   ├── train/                    # 4,487 images
│   ├── val/                      # 955 images
│   ├── test/                     # 981 images
│   ├── dataset_manifest.csv       # 6,423 entries
│   ├── split_metadata.json       # Split statistics
│   ├── qa_summary.json          # QA report
│   ├── split_distribution.png    # Visualization
│   └── species_distribution.png   # Visualization
└── species_to_family.json        # Label mapping
```

### C. Key References

- WoodID_plan.md — Project plan (PRIMARY)
- WoodID_Assignment_S2.docx — Research assignment (PRIMARY)
- conifer-family-classification.md — Design specification
- Ravindran et al. (2020) — XyloTron
- Lens et al. (2020) — Computer-assisted timber identification
- Filho et al. (2022) — Wood species classification from cross-section images

---

**Report Generated**: May 7, 2026  
**Author**: AI Assistant  
**Project**: WoodID-S2-2026
