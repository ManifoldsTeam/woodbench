---
phase: testing
title: Conifer Family Classification (Task A) — Testing Strategy
description: Testing approach, test cases, and quality assurance for family-level classification
---

# Testing Strategy — Task A: Conifer Family Classification

## Test Coverage Goals

### Overall Coverage Targets

| Type | Target | Rationale |
|------|--------|-----------|
| Unit Tests | 100% data/model modules | Core logic must be correct |
| Integration Tests | Critical paths + error handling | Training pipeline reliability |
| End-to-End Tests | Full training + evaluation | Validate complete workflow |
| Manual Testing | UI/visualization | Grad-CAM, plots need human check |

### Test Categories

| Category | Scope | Target Coverage |
|----------|-------|-----------------|
| Data Pipeline Tests | dataset.py, transforms.py, splitter.py | 100% |
| Model Tests | factory.py, heads.py | 100% |
| Training Tests | trainer.py, optimizer.py, callbacks.py | 80% |
| Evaluation Tests | metrics.py, confusion.py | 100% |
| Profiling Tests | efficiency.py | 100% |

---

## Unit Tests

### Data Module Tests

```python
# tests/test_data/test_dataset.py
import pytest
import torch
from pathlib import Path
from src.data.dataset import ConiferDataset

class TestConiferDataset:
    """Test suite for ConiferDataset."""
    
    @pytest.fixture
    def mock_dataset(self, tmp_path):
        """Create a mock dataset structure for testing."""
        # Create folder structure
        for split in ["train", "val", "test"]:
            for family in ["Cupressaceae", "Pinaceae"]:
                family_path = tmp_path / split / family
                species_path = family_path / "cunninghamia_lanceolata"
                species_path.mkdir(parents=True)
                
                # Create dummy image
                img = Image.new("RGB", (100, 100), color="red")
                img.save(species_path / "test.jpg")
        
        return tmp_path
    
    def test_dataset_length(self, mock_dataset):
        """Test that dataset returns correct number of samples."""
        dataset = ConiferDataset(mock_dataset, split="train")
        assert len(dataset) > 0
    
    def test_dataset_taskA_labels(self, mock_dataset):
        """Test Task A labels are family-level."""
        dataset = ConiferDataset(mock_dataset, split="train", task="taskA")
        assert dataset.classes == ConiferDataset.FAMILY_CLASSES
    
    def test_dataset_taskB_labels(self, mock_dataset):
        """Test Task B labels are species-level."""
        dataset = ConiferDataset(mock_dataset, split="train", task="taskB")
        assert "cunninghamia_lanceolata" in dataset.classes
    
    def test_getitem_returns_tensor(self, mock_dataset):
        """Test that __getitem__ returns correct types."""
        dataset = ConiferDataset(mock_dataset, split="train")
        image, label = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, int)
        assert image.shape[0] == 3  # RGB
    
    def test_transform_application(self, mock_dataset):
        """Test that transforms are applied correctly."""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        dataset = ConiferDataset(mock_dataset, split="train", transform=transform)
        image, _ = dataset[0]
        assert image.shape == (3, 224, 224)
    
    def test_invalid_split_raises(self, mock_dataset):
        """Test that invalid split raises error."""
        with pytest.raises(ValueError):
            ConiferDataset(mock_dataset, split="invalid")
    
    def test_class_distribution(self, mock_dataset):
        """Test that class distribution is balanced."""
        dataset = ConiferDataset(mock_dataset, split="train", task="taskA")
        from collections import Counter
        labels = [dataset[i][1] for i in range(len(dataset))]
        counts = Counter(labels)
        # Should have at least 2 classes
        assert len(counts) >= 2
```

### Transforms Module Tests

```python
# tests/test_data/test_transforms.py
import pytest
import torch
from src.data.transforms import get_train_transforms, get_val_transforms

class TestTransforms:
    """Test suite for data transforms."""
    
    def test_train_transforms_output_shape(self):
        """Test train transforms output correct shape."""
        transform = get_train_transforms(img_size=224)
        dummy_img = Image.new("RGB", (500, 500))
        result = transform(dummy_img)
        assert result.shape == (3, 224, 224)
    
    def test_train_transforms_output_type(self):
        """Test train transforms output tensor."""
        transform = get_train_transforms()
        dummy_img = Image.new("RGB", (100, 100))
        result = transform(dummy_img)
        assert isinstance(result, torch.Tensor)
    
    def test_val_transforms_output_shape(self):
        """Test val transforms output correct shape."""
        transform = get_val_transforms(img_size=224)
        dummy_img = Image.new("RGB", (500, 500))
        result = transform(dummy_img)
        assert result.shape == (3, 224, 224)
    
    def test_train_transforms_with_90rotation(self):
        """Test 90° rotation augmentation is included."""
        transform = get_train_transforms(use_90rotation=True)
        # Should not raise error
        dummy_img = Image.new("RGB", (100, 100))
        result = transform(dummy_img)
        assert result.shape == (3, 224, 224)
    
    def test_val_transforms_deterministic(self):
        """Test val transforms are deterministic (same output for same input)."""
        transform = get_val_transforms()
        dummy_img = Image.new("RGB", (100, 100))
        result1 = transform(dummy_img)
        result2 = transform(dummy_img)
        assert torch.allclose(result1, result2)
```

### Model Factory Tests

```python
# tests/test_models/test_factory.py
import pytest
import torch
from src.models.factory import create_model, BACKBONE_NAMES

class TestModelFactory:
    """Test suite for model creation."""
    
    @pytest.mark.parametrize("backbone", list(BACKBONE_NAMES.keys()))
    def test_create_model_output_shape(self, backbone):
        """Test model outputs correct shape."""
        model = create_model(backbone, num_classes=4)
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        assert output.shape == (1, 4)
    
    @pytest.mark.parametrize("backbone", list(BACKBONE_NAMES.keys()))
    def test_create_model_gradients(self, backbone):
        """Test that head has gradients, backbone is frozen."""
        model = create_model(backbone, num_classes=4, freeze_backbone=True)
        
        # Check head has gradients
        head_params = list(model.head.parameters())
        assert any(p.requires_grad for p in head_params)
        
        # Check backbone is frozen
        backbone_params = [p for n, p in model.named_parameters() if "head" not in n]
        assert all(not p.requires_grad for p in backbone_params)
    
    @pytest.mark.parametrize("backbone", list(BACKBONE_NAMES.keys()))
    def test_create_model_eval_mode(self, backbone):
        """Test model can switch to eval mode."""
        model = create_model(backbone)
        model.eval()
        assert not model.training
    
    def test_create_model_taskA_num_classes(self):
        """Test Task A (4-class) model creation."""
        model = create_model("resnet50", num_classes=4)
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        assert output.shape == (1, 4)
    
    def test_create_model_taskB_num_classes(self):
        """Test Task B (20-class) model creation."""
        model = create_model("resnet50", num_classes=20)
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        assert output.shape == (1, 20)
```

### Evaluation Metrics Tests

```python
# tests/test_eval/test_metrics.py
import pytest
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from src.eval.metrics import compute_metrics

class TestMetrics:
    """Test suite for evaluation metrics."""
    
    @pytest.fixture
    def mock_dataloader(self):
        """Create a mock dataloader with known labels."""
        # This would be replaced with actual mock
        pass
    
    def test_accuracy_perfect(self):
        """Test accuracy = 1.0 when all predictions correct."""
        # Create model that predicts correctly
        model = torch.nn.Linear(10, 4)
        # ... mock dataloader with known labels
        # metrics = compute_metrics(model, dataloader)
        # assert metrics["accuracy"] == 1.0
        pass
    
    def test_accuracy_zero(self):
        """Test accuracy = 0.0 when all predictions wrong."""
        pass
    
    def test_confusion_matrix_shape(self):
        """Test confusion matrix has correct shape."""
        # 4 families × 4 families for Task A
        y_true = [0, 1, 2, 3, 0]
        y_pred = [0, 1, 2, 0, 3]
        cm = confusion_matrix(y_true, y_pred)
        assert cm.shape == (4, 4)
    
    def test_per_class_f1(self):
        """Test per-class F1 calculation."""
        y_true = [0, 0, 1, 1, 2, 2, 3, 3]
        y_pred = [0, 0, 1, 0, 2, 2, 3, 3]
        from sklearn.metrics import f1_score
        f1_per_class = f1_score(y_true, y_pred, average=None)
        assert len(f1_per_class) == 4  # 4 classes
```

### Efficiency Profiling Tests

```python
# tests/test_profile/test_efficiency.py
import pytest
import torch
from src.profile.efficiency import profile_model

class TestEfficiency:
    """Test suite for efficiency profiling."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_profile_model_returns_dict(self):
        """Test profile_model returns expected keys."""
        model = torch.nn.Linear(100, 10)
        metrics = profile_model(model)
        
        expected_keys = ["params_millions", "flops_giga", "inference_time_ms", "model_size_mb"]
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], float)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_profile_model_inference_time_positive(self):
        """Test inference time is positive."""
        model = torch.nn.Linear(100, 10)
        metrics = profile_model(model)
        assert metrics["inference_time_ms"] > 0
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_profile_model_params_reasonable(self):
        """Test parameter count is reasonable."""
        model = torch.nn.Linear(100, 10)
        metrics = profile_model(model)
        # Small model should have < 1M params
        assert metrics["params_millions"] < 1.0
```

---

## Integration Tests

### Training Pipeline Integration

```python
# tests/test_integration/test_training_pipeline.py
import pytest
import torch
from pathlib import Path
from src.data.dataset import ConiferDataset, get_dataloaders
from src.models.factory import create_model
from src.train.trainer import TrainingEngine

class TestTrainingPipeline:
    """Integration tests for full training pipeline."""
    
    @pytest.fixture
    def mock_setup(self, tmp_path):
        """Setup mock dataset and model for integration testing."""
        # Create minimal mock dataset
        # Setup model
        # Return dataloaders and model
        pass
    
    def test_single_epoch_training(self, mock_setup):
        """Test that one epoch of training completes without error."""
        # model, train_loader, val_loader = mock_setup
        # engine = TrainingEngine(...)
        # metrics = engine.train_one_epoch(0)
        # assert "train_loss" in metrics
        # assert "train_acc" in metrics
        pass
    
    def test_validation_after_training(self, mock_setup):
        """Test validation metrics are computed."""
        # model, train_loader, val_loader = mock_setup
        # engine = TrainingEngine(...)
        # engine.train_one_epoch(0)
        # val_metrics = engine.validate()
        # assert "val_loss" in val_metrics
        # assert "val_acc" in val_metrics
        pass
    
    def test_checkpoint_saving(self, tmp_path):
        """Test that checkpoints are saved correctly."""
        # model = create_model("resnet50", num_classes=4)
        # engine = TrainingEngine(model, ...)
        # engine.save_checkpoint(tmp_path / "test.pth", {"val_acc": 95.0})
        # assert (tmp_path / "test.pth").exists()
        pass
    
    def test_checkpoint_loading(self, tmp_path):
        """Test that checkpoints can be loaded."""
        # Save and load checkpoint
        pass
```

### Data Split Integration

```python
# tests/test_integration/test_data_split.py

class TestDataSplit:
    """Test data splitting integrity."""
    
    def test_stratified_split_proportions(self):
        """Test that split creates correct proportions."""
        # Expected: 70% train, 15% val, 15% test
        # Check total images preserved
        pass
    
    def test_no_overlap_between_splits(self):
        """Test that train/val/test have no overlapping images."""
        pass
    
    def test_all_species_in_all_splits(self):
        """Test that all 20 species appear in all splits."""
        pass
```

### Multi-Seed Reproducibility

```python
# tests/test_integration/test_reproducibility.py

class TestReproducibility:
    """Test reproducibility across seeds."""
    
    def test_same_seed_same_results(self):
        """Test that same seed produces same results."""
        # Run with seed=42 twice, compare results
        pass
    
    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        # Run with seed=42 and seed=123, check different
        pass
    
    def test_multi_seed_mean_calculation(self):
        """Test that mean ± std is calculated correctly."""
        accuracies = [97.5, 97.8, 97.2]
        mean = sum(accuracies) / len(accuracies)
        variance = sum((x - mean) ** 2 for x in accuracies) / len(accuracies)
        std = variance ** 0.5
        assert 97.1 < mean < 97.9
        assert 0.2 < std < 0.5
```

---

## End-to-End Tests

### Complete Training Run

```python
# tests/test_e2e/test_full_training.py

class TestFullTraining:
    """End-to-end tests for complete training."""
    
    @pytest.mark.slow
    def test_resnet50_full_training(self):
        """Test complete training of ResNet-50 on Task A."""
        # This is a slow test - run only with --runslow flag
        # 1. Setup dataset
        # 2. Create model
        # 3. Train for 50 epochs
        # 4. Evaluate on test set
        # 5. Check accuracy > 97%
        pass
    
    @pytest.mark.slow
    def test_all_models_trainable(self):
        """Test that all 5 models can complete training."""
        # Run short training (5 epochs) for each model
        # Check no errors
        pass
```

### Inference Pipeline

```python
# tests/test_e2e/test_inference.py

class TestInference:
    """End-to-end tests for inference pipeline."""
    
    def test_single_image_inference(self):
        """Test inference on single image."""
        # Load checkpoint
        # Load image
        # Run inference
        # Check output shape and class
        pass
    
    def test_batch_inference(self):
        """Test inference on batch of images."""
        # Load checkpoint
        # Run inference on batch
        # Check batch dimension
        pass
    
    def test_gradcam_generation(self):
        """Test Grad-CAM visualization generation."""
        # Load checkpoint
        # Generate Grad-CAM for one image
        # Check output shape (H, W, 3)
        pass
```

---

## Test Data

### Mock Data Fixtures

```python
# tests/conftest.py
import pytest
import torch
from pathlib import Path

@pytest.fixture
def mock_conifer_dataset(tmp_path):
    """Create a minimal mock Conifer dataset for testing."""
    # Create structure
    # data/train/{family}/{species}/*.jpg
    # data/val/{family}/{species}/*.jpg
    # data/test/{family}/{species}/*.jpg
    return tmp_path

@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, 3),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(16, 4),
    )
    return model

@pytest.fixture
def mock_dataloader():
    """Create a mock dataloader."""
    # Return pre-configured dataloader with fake data
    pass
```

---

## Test Reporting & Coverage

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run only unit tests (fast)
pytest tests/ -m "not slow"

# Run only slow tests (full training)
pytest tests/ -m slow --runslow

# Run specific test file
pytest tests/test_data/test_dataset.py -v

# Run with verbose output
pytest tests/ -vv -s
```

### Coverage Targets

| Module | Coverage Target | Critical Functions |
|--------|----------------|---------------------|
| data/dataset.py | 100% | `_collect_samples`, `__getitem__` |
| data/transforms.py | 100% | `get_train_transforms`, `get_val_transforms` |
| models/factory.py | 100% | `create_model` |
| train/trainer.py | 80% | `train_one_epoch`, `validate`, `fit` |
| eval/metrics.py | 100% | `compute_metrics` |
| profile/efficiency.py | 100% | `profile_model` |

---

## Manual Testing

### Grad-CAM Visualization Check

| Checkpoint | Item | Description |
|------------|------|-------------|
| [ ] | Cupressaceae sample 1 | Heatmap shows wood ring patterns |
| [ ] | Cupressaceae sample 2 | Consistent attention region |
| [ ] | Pinaceae sample 1 | Heatmap shows tracheid structure |
| [ ] | Pinaceae sample 2 | Consistent attention region |
| [ ] | Podocarpaceae sample | Different pattern from Cupressaceae |
| [ ] | Taxaceae sample | Different pattern from others |
| [ ] | Model comparison | All 5 models show similar regions |

### Confusion Matrix Check

| Checkpoint | Item | Description |
|------------|------|-------------|
| [ ] | Diagonal values | Should be > 95% |
| [ ] | Off-diagonal values | Cupressaceae vs Pinaceae confusion expected |
| [ ] | Per-family F1 | All > 0.95 |

### Efficiency Results Check

| Checkpoint | Item | Description |
|------------|------|-------------|
| [ ] | MobileNetV3-L inference | Should be < 10ms/image |
| [ ] | Swin-T inference | Should be < 30ms/image |
| [ ] | Model size ratio | MobileNetV3 < ConvNeXt < Swin |

---

## Performance Testing

### Training Performance Benchmarks

| Model | Expected Time (RTX 3080) | Target | Pass/Fail |
|-------|---------------------------|--------|-----------|
| ResNet-50 | ~2 hours | < 3 hours | |
| EfficientNet-B3 | ~2.5 hours | < 4 hours | |
| MobileNetV3-L | ~1.5 hours | < 2 hours | |
| ConvNeXt-Tiny | ~2 hours | < 3 hours | |
| Swin-T | ~2.5 hours | < 4 hours | |

### Inference Performance Benchmarks

| Model | Target Time | Pass/Fail |
|-------|-------------|-----------|
| MobileNetV3-L | < 10ms | |
| ResNet-50 | < 15ms | |
| EfficientNet-B3 | < 20ms | |
| ConvNeXt-Tiny | < 25ms | |
| Swin-T | < 30ms | |

---

## Bug Tracking

### Known Issues Log

| Issue ID | Description | Severity | Status | Workaround |
|----------|-------------|----------|--------|------------|
| TBD | ... | ... | ... | ... |

### Bug Severity Levels

| Level | Description | Response Time |
|-------|-------------|---------------|
| Critical | Training fails completely | Immediate |
| High | Accuracy < 90% or wrong metrics | 1 day |
| Medium | Minor UI/plot issues | 1 week |
| Low | Cosmetic issues | Next sprint |

---

## Regression Testing Strategy

### Before Each Commit

- Run unit tests: `pytest tests/test_data tests/test_models -v`
- Quick smoke test: `python scripts/train_taskA.py --epochs 1 --debug`

### Before Each Experiment Run

- Verify dataset integrity: `python scripts/verify_dataset.py`
- Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`

### After Each Experiment

- Run evaluation on test set
- Check metrics are in expected range
- Verify checkpoint saved correctly

### Weekly

- Full test suite: `pytest tests/ --cov=src`
- Compare with previous week's results
