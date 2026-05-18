# Implementation Plan: Task A Augmentation Pipeline

## Overview

Triển khai pipeline augmentation offline cho 2 lớp thiểu số (Podocarpaceae, Taxaceae) trong Task A. Pipeline sinh ảnh augmented dưới dạng file thực tại `data/task-A/`, tạo notebook Colab-compatible, và upload lên HuggingFace. Implementation sử dụng Python với `albumentations`, `PIL/Pillow`, và `huggingface_hub`.

## Tasks

- [x] 1. Tạo module augmentation pipeline và core interfaces
  - [x] 1.1 Tạo file `src/data/augmentation.py` với `AugmentationMethod` enum và `AugmentationPipeline` class
    - Định nghĩa enum `AugmentationMethod` với 7 methods: HFLIP, VFLIP, ROT180, ROT270, CROP_RESIZE, COLOR_JITTER, TRANSPOSE
    - Implement `__init__` với params: seed=42, target_count=(1500, 1700), crop_min_area=0.7, color_params
    - Implement `get_transform(method)` trả về `A.Compose` tương ứng với từng method
    - Implement `augment_image(image, method)` áp dụng transform lên numpy array
    - Implement `compute_augmentation_plan(original_count, target_count)` phân phối đều methods theo round-robin
    - Sử dụng albumentations transforms với params từ design: HFlip(p=1.0), VFlip(p=1.0), Rotate(limit=(180,180)), Rotate(limit=(270,270)), RandomResizedCrop(height=2048, width=2560, scale=(0.7, 0.9)), ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05), Transpose(p=1.0)
    - Vietnamese comments, no emoji
    - _Requirements: 2.1, 2.2, 2.3, 2.5, 2.6, 3.1, 3.2, 3.3, 6.4_

  - [ ]* 1.2 Write property test cho AugmentationPipeline (Property 4: Resolution preservation)
    - **Property 4: Resolution preservation**
    - **Validates: Requirements 2.5, 2.6**
    - Dùng Hypothesis generate random numpy arrays (2048x2560x3) và random AugmentationMethod
    - Verify output dimensions luôn là 2560x2048 (width x height)

  - [ ]* 1.3 Write property test cho augmentation plan (Property 5: Uniform method distribution)
    - **Property 5: Uniform method distribution**
    - **Validates: Requirements 3.1**
    - Dùng Hypothesis generate random original_count (100-300) và target_count (1500-1700)
    - Verify deviation per method <= 15% from mean

- [x] 2. Implement DirectoryCopier và ManifestWriter
  - [x] 2.1 Tạo file `src/data/directory_copier.py` với class `DirectoryCopier`
    - Implement `copy_split(src_root, dst_root, split, families)` copy real files (không symlink)
    - Resolve symlinks khi copy (source data dùng symlinks tới data/raw/)
    - Trả về số file đã copy
    - Vietnamese comments, no emoji
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

  - [x] 2.2 Tạo file `src/data/manifest_writer.py` với class `ManifestWriter`
    - Implement `__init__(output_path)`, `add_entry(original_path, augmented_path, method, family, species)`, `save()`
    - Output CSV với columns: original_path, augmented_path, method, family, species
    - Vietnamese comments, no emoji
    - _Requirements: 6.1, 6.2_

  - [x] 2.3 Write property test cho DirectoryCopier (Property 1: No symlinks in output)
    - **Property 1: No symlinks in output**
    - **Validates: Requirements 1.2**
    - Tạo temp directory với mix of symlinks và real files, verify output không có symlinks

  - [x] 2.4 Write property test cho file copy integrity (Property 2: Non-augmented files are byte-identical)
    - **Property 2: Non-augmented files are byte-identical copies**
    - **Validates: Requirements 1.3, 1.4**
    - Verify copied files byte-identical với source (sau khi resolve symlinks)

- [x] 3. Checkpoint - Verify core modules
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Tạo notebook `notebooks/phase 0/3_task_a_augmentation.ipynb`
  - [x] 4.1 Tạo Cell 1: Root Detection (local-only) + Cell 2: Verify source data exists
    - Cell 1: phát hiện ROOT bằng cách đi ngược lên cây thư mục từ vị trí notebook đến khi gặp marker (`.kiro/` hoặc `requirements.txt`); fallback `Path.cwd().parent.parent`. Insert ROOT vào `sys.path`.
    - Cell 2: sanity-check `SRC_DIR = ROOT / "data" / "split"`, raise `FileNotFoundError` với thông điệp tiếng Việt nếu thiếu (yêu cầu chạy `data preparation.ipynb` trước), liệt kê các splits + families hiện có.
    - Sử dụng ROOT-based paths, Vietnamese comments, no emoji
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

  - [x] 4.2 Tạo Cell 4: Import + Config
    - Import AugmentationPipeline, DirectoryCopier, ManifestWriter từ src/data/
    - Định nghĩa config: SEED=42, TARGET_COUNT=1600, MINORITY_FAMILIES, MAJORITY_FAMILIES
    - Định nghĩa SRC_DIR = ROOT / "data" / "split", DST_DIR = ROOT / "data" / "task-A"
    - _Requirements: 4.4, 4.5, 4.6, 6.4_

  - [x] 4.3 Tạo Cell 5: Directory setup + copy engine
    - Tạo cấu trúc thư mục `data/task-A/{split}/{family}/{species}/`
    - Copy val/test splits nguyên bản (all families)
    - Copy train majority classes nguyên bản
    - Copy train minority classes originals
    - Hiển thị progress và số file đã copy
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

  - [x] 4.4 Tạo Cell 6: Augmentation pipeline execution
    - Instantiate AugmentationPipeline với seed=42
    - Compute augmentation plan cho mỗi minority class
    - Execute augmentation: đọc ảnh gốc, apply transform, save augmented file
    - Naming convention: `{original_stem}_aug{method}_{idx}.png`
    - Hiển thị progress per family
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 3.1, 3.2, 3.3, 3.4_

  - [x] 4.5 Tạo Cell 7-8: Manifest CSV generation + QA Summary
    - Cell 7: Ghi manifest CSV tại `data/task-A/augmentation_manifest.csv`
    - Cell 8: Hiển thị summary table (per-family counts before/after), method distribution, class balance ratios
    - _Requirements: 3.5, 6.1, 6.2, 6.3_

  - [x] 4.6 Tạo Cell 9: Upload to HuggingFace
    - Upload `data/task-A/` lên `ny0g507h3p/woodbench_A` dùng `huggingface_hub.upload_folder`
    - Handle authentication errors với clear message
    - Hiển thị upload progress và confirmation URL
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [ ] 5. Checkpoint - Verify notebook structure
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. Property tests và integration tests
  - [ ]* 6.1 Write property test cho augmented file naming (Property 3: Naming and method validity)
    - **Property 3: Augmented file naming and method validity**
    - **Validates: Requirements 2.2, 2.3, 2.4**
    - Verify tất cả augmented filenames match pattern `{stem}_aug{method}_{idx}.png` với method in valid set

  - [ ]* 6.2 Write property test cho PNG format (Property 6: PNG format validity)
    - **Property 6: PNG format validity**
    - **Validates: Requirements 3.4**
    - Verify augmented files có extension .png và readable by PIL without error

  - [ ]* 6.3 Write property test cho manifest completeness (Property 7: Manifest completeness)
    - **Property 7: Manifest completeness**
    - **Validates: Requirements 6.2**
    - Cross-reference filesystem augmented files với manifest CSV entries

  - [ ]* 6.4 Write property test cho reproducibility (Property 8: Determinism)
    - **Property 8: Reproducibility (deterministic output)**
    - **Validates: Requirements 6.4**
    - Run augmentation twice với seed=42 trên random subset, verify byte-identical outputs

  - [ ]* 6.5 Write unit tests cho notebook structure validation
    - Verify Root Detection cell present
    - Verify Root Detection cell uses local-only marker walk (`.kiro/` or `requirements.txt`)
    - Verify no hard-coded paths
    - Verify Vietnamese comments, no emoji
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

- [ ] 7. Wire components và final integration
  - [ ] 7.1 Update `src/data/__init__.py` exports
    - Export AugmentationMethod, AugmentationPipeline, DirectoryCopier, ManifestWriter
    - Đảm bảo `from src.data import ...` hoạt động sau sys.path.insert
    - _Requirements: 4.5_

  - [ ] 7.2 Verify end-to-end notebook execution trên small subset
    - Tạo test script chạy pipeline trên 5 images per minority class
    - Verify full workflow: copy -> augment -> manifest -> summary
    - _Requirements: 1.1, 2.1, 3.5, 6.1_

- [ ] 8. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from design document (8 properties)
- Unit tests validate specific examples and edge cases
- Implementation language: Python (albumentations, PIL/Pillow, huggingface_hub)
- Conda environment: `wood`
- Vietnamese comments throughout, no emoji in code
- Source data at `data/split/` uses symlinks to `data/raw/` — DirectoryCopier must resolve these

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "2.1", "2.2"] },
    { "id": 1, "tasks": ["1.2", "1.3", "2.3", "2.4"] },
    { "id": 2, "tasks": ["4.1", "4.2", "7.1"] },
    { "id": 3, "tasks": ["4.3", "4.4"] },
    { "id": 4, "tasks": ["4.5", "4.6"] },
    { "id": 5, "tasks": ["6.1", "6.2", "6.3", "6.4", "6.5"] },
    { "id": 6, "tasks": ["7.2"] }
  ]
}
```
