# Requirements Document

## Introduction

Task A (family-level classification, 4 classes) trong dự án WoodID-S2-2026 gặp vấn đề mất cân bằng lớp nghiêm trọng: Podocarpaceae (217 train, 4.8%) và Taxaceae (206 train, 4.6%) chỉ chiếm ~9.4% tổng dữ liệu huấn luyện, trong khi Cupressaceae (2,398 train, 53.4%) và Pinaceae (1,666 train, 37.1%) chiếm hơn 90%. Kết quả thực nghiệm cho thấy Taxaceae chỉ đạt 70.27% accuracy với ResNet50, trong khi các lớp đa số đạt >97%.

Feature này tạo pipeline augmentation offline cho 2 lớp thiểu số, sinh ảnh augmented dưới dạng file thực (real copies) tại `data/task-A/`, đồng thời tạo notebook Colab-compatible để thực thi pipeline và đẩy kết quả lên HuggingFace.

## Glossary

- **Augmentation_Pipeline**: Module Python thực hiện việc sinh ảnh augmented từ ảnh gốc bằng các phép biến đổi hình ảnh
- **Minority_Class**: Lớp có số lượng mẫu huấn luyện thấp hơn đáng kể so với lớp đa số (Podocarpaceae: 217, Taxaceae: 206)
- **Majority_Class**: Lớp có số lượng mẫu huấn luyện lớn (Cupressaceae: 2,398; Pinaceae: 1,666)
- **Target_Count**: Số lượng mẫu mục tiêu sau augmentation cho mỗi lớp thiểu số, nằm trong khoảng 1,500-1,700 (tương đương mức Pinaceae)
- **Augmentation_Method**: Một phép biến đổi hình ảnh cụ thể (HFlip, VFlip, Rotation 180/270, RandomCrop+Resize, ColorJitter, Transpose)
- **Isotropic_Image**: Ảnh mặt cắt ngang gỗ không có hướng tự nhiên, cho phép áp dụng rotation/flip mà không mất thông tin ngữ nghĩa
- **Task_A_Directory**: Thư mục `data/task-A/` chứa dataset đã augment với cấu trúc `{split}/{family}/{species}/`
- **Augmented_File**: File ảnh được sinh từ ảnh gốc, đặt tên theo quy ước `{original_filename}_aug{method}_{idx}.ext`
- **Notebook**: File Jupyter notebook tại `notebooks/phase 0/3_task_a_augmentation.ipynb` chạy local trong conda env `wood` cho phase 0
- **HuggingFace_Repo**: Repository `ny0g507h3p/woodbench_A` trên HuggingFace Hub để lưu trữ dataset Task A

## Requirements

### Requirement 1: Tạo cấu trúc thư mục Task A

**User Story:** As a researcher, I want to have a dedicated directory structure for Task A data, so that augmented and original data are organized separately from the raw split data.

#### Acceptance Criteria

1. WHEN the Augmentation_Pipeline is executed, THE Task_A_Directory SHALL create the folder structure `data/task-A/{split}/{family}/{species}/` for all 4 families and all 3 splits (train, val, test)
2. THE Task_A_Directory SHALL contain real file copies (not symlinks) of all images
3. WHEN copying images for val and test splits, THE Augmentation_Pipeline SHALL copy original files without applying any augmentation
4. WHEN copying images for the train split of Majority_Class families (Cupressaceae, Pinaceae), THE Augmentation_Pipeline SHALL copy original files without applying any augmentation

### Requirement 2: Augmentation cho lớp thiểu số

**User Story:** As a researcher, I want to augment minority class images to reach a target count matching Pinaceae level, so that class imbalance is reduced for Task A training.

#### Acceptance Criteria

1. WHEN augmenting Minority_Class images, THE Augmentation_Pipeline SHALL generate Augmented_File images until each Minority_Class reaches the Target_Count of 1,500 to 1,700 samples (including originals)
2. THE Augmentation_Pipeline SHALL apply only the following Augmentation_Method transforms: HFlip, VFlip, Rotation(180), Rotation(270), RandomCrop+Resize, ColorJitter, Transpose
3. THE Augmentation_Pipeline SHALL NOT apply Rotation(90) as an augmentation method for class balancing (reserved for ablation study)
4. WHEN generating an Augmented_File, THE Augmentation_Pipeline SHALL name the file following the convention `{original_filename}_aug{method}_{idx}.{ext}`
5. THE Augmentation_Pipeline SHALL preserve the original image resolution (2560x2048) in augmented outputs without resizing (resize is handled by the training pipeline)
6. IF RandomCrop+Resize is applied, THEN THE Augmentation_Pipeline SHALL crop a random region and resize it back to the original resolution (2560x2048)

### Requirement 3: Đảm bảo chất lượng augmentation

**User Story:** As a researcher, I want augmented images to maintain quality and diversity, so that the model learns meaningful features rather than artifacts.

#### Acceptance Criteria

1. THE Augmentation_Pipeline SHALL distribute augmentation methods uniformly across all Augmented_File images for each Minority_Class to ensure method diversity
2. WHEN applying ColorJitter, THE Augmentation_Pipeline SHALL use moderate parameter ranges (brightness, contrast, saturation, hue) that preserve wood grain visibility
3. WHEN applying RandomCrop+Resize, THE Augmentation_Pipeline SHALL crop at least 70% of the original image area to preserve sufficient anatomical detail
4. THE Augmentation_Pipeline SHALL produce Augmented_File images in the same format (PNG) as the original images
5. WHEN the Augmentation_Pipeline completes, THE Notebook SHALL display a summary table showing per-family sample counts before and after augmentation for the train split

### Requirement 4: Notebook cho phase 0 (local execution)

**User Story:** As a researcher, I want a phase-0 notebook that executes the full augmentation pipeline locally in the conda `wood` environment, so that I can produce the augmented dataset artifact deterministically before later Colab training phases.

#### Acceptance Criteria

1. THE Notebook SHALL be created at path `notebooks/phase 0/3_task_a_augmentation.ipynb`
2. THE Notebook SHALL include a Root Detection cell as the first code cell that detects the repository root from the notebook's location (by walking parent directories until a marker such as the `.kiro` directory or `requirements.txt` is found) and inserts it into `sys.path`
3. THE Notebook SHALL include a sanity-check cell that verifies the source dataset exists at `data/split/`, raises a clear error message instructing the user to run the phase-0 data preparation notebook if missing, and lists the present splits and families when found
4. THE Notebook SHALL use `ROOT`-based paths for all file references without any hard-coded relative paths
5. THE Notebook SHALL include Vietnamese comments and documentation following the project language convention
6. THE Notebook SHALL NOT contain any emoji or icon characters in code, comments, or documentation

### Requirement 5: Upload artifact len HuggingFace cho cac phase sau

**User Story:** As a researcher, I want the phase-0 notebook to push the augmented Task A dataset to HuggingFace as the end-of-phase-0 artifact, so that subsequent Colab training phases can pull it without re-running augmentation locally.

#### Acceptance Criteria

1. WHEN augmentation is complete, THE Notebook SHALL upload the entire `data/task-A/` directory to the HuggingFace_Repo (`ny0g507h3p/woodbench_A`) as the final phase-0 step
2. THE Notebook SHALL use the `huggingface_hub` library for uploading with appropriate authentication handling
3. IF the upload fails due to authentication error, THEN THE Notebook SHALL display a clear error message instructing the user to run `huggingface-cli login`
4. THE Notebook SHALL display upload progress and final confirmation with the repository URL upon successful upload

### Requirement 6: Logging và truy xuất nguồn gốc

**User Story:** As a researcher, I want to track which augmentation methods were applied to generate each file, so that I can trace data provenance and reproduce results.

#### Acceptance Criteria

1. WHEN the Augmentation_Pipeline completes, THE Notebook SHALL generate a manifest CSV file at `data/task-A/augmentation_manifest.csv` containing columns: original_path, augmented_path, method, family, species
2. THE manifest CSV SHALL include entries for all Augmented_File images generated during the pipeline execution
3. WHEN the Augmentation_Pipeline completes, THE Notebook SHALL log a summary including: total images generated per class, augmentation method distribution, and final class balance ratios
4. THE Augmentation_Pipeline SHALL use a fixed random seed (42) to ensure reproducibility of RandomCrop and ColorJitter transforms
