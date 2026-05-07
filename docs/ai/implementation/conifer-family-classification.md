---
phase: implementation
title: Conifer Family Classification (Task A) — Implementation Guide
description: Technical implementation notes, code patterns, and development setup
---

# Implementation Guide — Task A: Conifer Family Classification

## Development Setup

### Prerequisites

```bash
# Python version
python >= 3.10

# NVIDIA driver + CUDA
nvidia-smi  # Check CUDA 12.x available
nvcc --version  # Should show 12.x
```

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install timm albumentations scikit-learn matplotlib seaborn pandas numpy
pip install ptflops thop  # For FLOPs calculation

# Optional: Experiment tracking
pip install wandb  # or tensorboard

# Optional: Grad-CAM
pip install grad-cam
```

### Environment Variables

```bash
# .env file
export DATA_ROOT=/path/to/IC4SD-Wood-Conifer20v1
export OUTPUT_DIR=/path/to/woodbench/experiments/taskA
export WANDB_PROJECT=woodid-taskA
export CUDA_VISIBLE_DEVICES=0
```

### Verify Setup

```python
# src/verify_setup.py
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

---

## Code Structure

### Directory Layout

```
woodbench/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py      # ConiferDataset, get_dataloaders
│   │   ├── transforms.py   # Augmentation pipelines
│   │   └── splitter.py     # Stratified split logic
│   ├── models/
│   │   ├── __init__.py
│   │   ├── factory.py      # create_model()
│   │   └── heads.py        # ClassificationHead
│   ├── train/
│   │   ├── __init__.py
│   │   ├── trainer.py      # TrainingEngine class
│   │   ├── optimizer.py    # create_optimizer_scheduler()
│   │   └── callbacks.py    # CheckpointCallback
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py      # compute_metrics()
│   │   ├── confusion.py    # plot_confusion_matrix()
│   │   └── gradcam.py      # GradCAMGenerator
│   ├── profile/
│   │   ├── __init__.py
│   │   └── efficiency.py   # profile_model()
│   └── utils/
│       ├── __init__.py
│       ├── seed.py         # set_seed()
│       └── logger.py       # setup_logger()
├── configs/
│   ├── taskA_baseline.yaml
│   └── taskA_90rotation.yaml
├── scripts/
│   ├── train_taskA.py       # Main training script
│   ├── eval_taskA.py        # Evaluation script
│   ├── profile_all.py       # Efficiency profiling
│   └── run_ablation.py      # Ablation study
├── experiments/
│   └── taskA/
│       └── {model_name}/
│           ├── seed_42/
│           ├── seed_123/
│           └── seed_456/
└── requirements.txt
```

---

## Core Features Implementation

### 1. Dataset Module

#### Dataset Class

```python
# src/data/dataset.py
from pathlib import Path
from typing import Optional, Callable, Tuple, List

class ConiferDataset(torch.utils.data.Dataset):
    """
    Dataset for Conifer wood classification.
    
    Structure: data/{split}/{family}/{species}/images.jpg
    
    Args:
        data_root: Root directory containing train/val/test folders
        split: 'train' | 'val' | 'test'
        transform: Optional transform to apply
        task: 'taskA' (family) or 'taskB' (species)
        family_to_idx: Optional mapping of family names to indices
    """
    
    FAMILY_CLASSES = [
        "Cupressaceae",
        "Pinaceae", 
        "Podocarpaceae",
        "Taxaceae"
    ]
    
    def __init__(
        self,
        data_root: Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        task: str = "taskA",
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.task = task
        
        # Build class-to-index mapping
        if task == "taskA":
            self.classes = self.FAMILY_CLASSES
        else:
            self.classes = self._get_species_classes()
            
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Collect samples
        self.samples = self._collect_samples()
    
    def _collect_samples(self) -> List[Tuple[Path, int]]:
        """Walk directory tree and collect (image_path, label) tuples."""
        samples = []
        split_path = self.data_root / self.split
        
        for family_folder in split_path.iterdir():
            if not family_folder.is_dir():
                continue
            
            if self.task == "taskA":
                label = self.class_to_idx[family_folder.name]
                for species_folder in family_folder.iterdir():
                    if species_folder.is_dir():
                        for img_path in species_folder.glob("*.jpg"):
                            samples.append((img_path, label))
            else:
                # Task B: species as label
                for species_folder in family_folder.iterdir():
                    if species_folder.is_dir():
                        label = self.class_to_idx[species_folder.name]
                        for img_path in species_folder.glob("*.jpg"):
                            samples.append((img_path, label))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
```

#### DataLoaders

```python
# src/data/dataset.py (continued)
def get_dataloaders(
    data_root: Path,
    batch_size: int = 64,
    num_workers: int = 4,
    use_90rotation: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders with appropriate transforms.
    
    Args:
        data_root: Path to dataset
        batch_size: Batch size
        num_workers: DataLoader workers
        use_90rotation: Whether to use 90° rotation augmentation
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_transform = get_train_transforms(use_90rotation=use_90rotation)
    val_transform = get_val_transforms()
    
    train_dataset = ConiferDataset(data_root, split="train", transform=train_transform)
    val_dataset = ConiferDataset(data_root, split="val", transform=val_transform)
    test_dataset = ConiferDataset(data_root, split="test", transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader
```

### 2. Transforms Module

```python
# src/data/transforms.py
from torchvision import transforms

def get_train_transforms(
    img_size: int = 224,
    use_90rotation: bool = False,
) -> transforms.Compose:
    """Get training augmentation pipeline."""
    
    aug_list = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),  # Isotropic - valid for cross-section
    ]
    
    if use_90rotation:
        # Discrete 90° rotation augmentation for isotropic cross-section images
        # Hypothesis: 0°, 90°, 180°, 270° rotations preserve semantics
        # Randomly select one of 4 discrete orientations with equal probability
        aug_list.append(transforms.RandomChoice([
            transforms.Lambda(lambda x: transforms.functional.rotate(x, 0)),
            transforms.Lambda(lambda x: transforms.functional.rotate(x, 90)),
            transforms.Lambda(lambda x: transforms.functional.rotate(x, 180)),
            transforms.Lambda(lambda x: transforms.functional.rotate(x, 270)),
        ]))
    else:
        aug_list.append(transforms.RandomRotation(30))  # Continuous rotation ±30°
    
    aug_list.extend([
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    return transforms.Compose(aug_list)

def get_val_transforms(img_size: int = 224) -> transforms.Compose:
    """Get validation/test transform (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
```

### 3. Model Factory

```python
# src/models/factory.py
import timm
import torch.nn as nn

def create_model(
    backbone_name: str,
    num_classes: int = 4,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    """
    Create model with pretrained backbone and classification head.
    
    Args:
        backbone_name: timm model name
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
        freeze_backbone: Freeze backbone parameters
    
    Returns:
        Model with backbone + head
    """
    # Create model
    model = timm.create_model(
        backbone_name,
        pretrained=pretrained,
        num_classes=0,  # Remove classifier
    )
    
    # Get feature dimension
    in_features = model.num_features
    
    # Create new classifier head
    model.head = nn.Linear(in_features, num_classes)
    
    # Freeze backbone if specified
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        # Ensure head is trainable
        for param in model.head.parameters():
            param.requires_grad = True
    
    return model

# Backbone name mapping
BACKBONE_NAMES = {
    "resnet50": "resnet50",
    "efficientnet_b3": "tf_efficientnet_b3",
    "mobilenetv3_large": "mobilenetv3_large_100",
    "convnext_tiny": "convnext_tiny",
    "swin_tiny": "swin_tiny_patch4_window7_224",
}
```

### 4. Training Loop

```python
# src/train/trainer.py
from typing import Dict, Optional
from pathlib import Path
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

class TrainingEngine:
    """Training engine for Task A."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion: nn.Module,
        device: str = "cuda",
        use_amp: bool = True,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.use_amp = use_amp
        self.scaler = GradScaler() if use_amp else None
        
        self.model.to(device)
    
    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return {
            "train_loss": total_loss / len(self.train_loader),
            "train_acc": 100.0 * correct / total,
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return {
            "val_loss": total_loss / len(self.val_loader),
            "val_acc": 100.0 * correct / total,
        }
    
    def fit(
        self,
        epochs: int,
        output_dir: Path,
        name: str = "model",
    ) -> Dict[str, float]:
        """Train for multiple epochs with checkpointing."""
        best_acc = 0
        
        for epoch in range(epochs):
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate()
            
            self.scheduler.step()
            
            # Log metrics
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}, Acc: {train_metrics['train_acc']:.2f}%")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}, Acc: {val_metrics['val_acc']:.2f}%")
            
            # Save checkpoint
            if val_metrics["val_acc"] > best_acc:
                best_acc = val_metrics["val_acc"]
                self.save_checkpoint(output_dir / f"{name}_best.pth", val_metrics)
            
            # Save last
            self.save_checkpoint(output_dir / f"{name}_last.pth", val_metrics)
        
        return {"best_val_acc": best_acc}
    
    def save_checkpoint(self, path: Path, metrics: Dict):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
        }, path)
```

### 5. Evaluation Metrics

```python
# src/eval/metrics.py
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
import torch
from typing import Dict

@torch.no_grad()
def compute_metrics(
    model: torch.nn.Module,
    data_loader,
    device: str = "cuda",
    num_classes: int = 4,
) -> Dict[str, float]:
    """Compute all evaluation metrics."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    for images, labels in data_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision_macro": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall_macro": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(all_labels, all_preds),
    }
    
    # Per-class F1
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    metrics["per_class_f1"] = {
        f"class_{i}": float(f1) for i, f1 in enumerate(per_class_f1)
    }
    
    return metrics

@torch.no_grad()
def measure_inference_time(
    model: torch.nn.Module,
    input_size: tuple = (3, 224, 224),
    device: str = "cuda",
    num_runs: int = 100,
    warmup_runs: int = 10,
) -> float:
    """Measure average inference time per image (ms)."""
    model.eval()
    
    # Dummy input
    dummy_input = torch.randn(1, *input_size).to(device)
    
    # Warmup
    for _ in range(warmup_runs):
        _ = model(dummy_input)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Timed runs
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    times = []
    for _ in range(num_runs):
        start.record()
        _ = model(dummy_input)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    return np.mean(times)
```

### 6. Grad-CAM

```python
# src/eval/gradcam.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

class GradCAMGenerator:
    """Generate Grad-CAM visualizations for model interpretation."""
    
    TARGET_LAYERS = {
        "resnet50": "layer4[-1].conv3",
        "efficientnet_b3": "features[-1]",
        "mobilenetv3_large": "features[-1]",
        "convnext_tiny": "features[-1]",
        "swin_tiny": "layers[-1].blocks[-1]",
    }
    
    def __init__(self, model: torch.nn.Module, backbone_name: str):
        self.model = model
        self.backbone_name = backbone_name
        self.target_layer = self._get_target_layer()
        self.cam = GradCAM(model=self.model, target_layers=[self.target_layer])
    
    def _get_target_layer(self):
        """Get the target layer for Grad-CAM based on backbone."""
        if self.backbone_name == "resnet50":
            return self.model.layer4[-1].conv3
        elif "efficientnet" in self.backbone_name:
            return self.model.features[-1]
        elif "mobilenet" in self.backbone_name:
            return self.model.features[-1]
        elif "convnext" in self.backbone_name:
            return self.model.features[-1]
        elif "swin" in self.backbone_name:
            return self.model.layers[-1].blocks[-1].norm1
    
    @torch.no_grad()
    def generate(
        self,
        image: torch.Tensor,
        target_category: int = None,
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap for an image."""
        # image: (C, H, W) tensor, normalized
        # Returns: (H, W, 3) uint8 image with heatmap overlay
        grayscale_cam = self.cam(input_tensor=image.unsqueeze(0), target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]
        
        # Denormalize image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_denorm = image * std + mean
        img_np = img_denorm.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)
        
        # Overlay heatmap
        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        
        return visualization
```

---

## Patterns & Best Practices

### Reproducibility

```python
# src/utils/seed.py
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic options (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
```

### Checkpoint Structure

```python
# Checkpoint format
checkpoint = {
    "epoch": 50,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "metrics": {"val_acc": 98.5, "val_f1": 98.3},
    "config": {
        "backbone": "resnet50",
        "task": "taskA",
        "num_classes": 4,
        "batch_size": 64,
        "lr": 1e-4,
    },
    "seed": 42,
}
```

### Logging

```python
# src/utils/logger.py
import logging
from pathlib import Path

def setup_logger(name: str, log_dir: Path) -> logging.Logger:
    """Setup logger with file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler
    log_file = log_dir / f"{name}.log"
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger
```

---

## Error Handling

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| CUDA OOM | Batch size too large | Reduce batch_size to 32 or 16 |
| NaN loss | Learning rate too high | Reduce LR to 1e-5 |
| Low accuracy | Data path incorrect | Verify folder structure |
| Slow training | num_workers=0 | Set num_workers=4 |

### Debug Commands

```bash
# Check dataset
python -c "from src.data.dataset import ConiferDataset; d = ConiferDataset('data'); print(len(d))"

# Check model
python -c "from src.models.factory import create_model; m = create_model('resnet50'); print(m)"

# Test training step
python scripts/train_taskA.py --backbone resnet50 --epochs 1 --debug
```

---

## Performance Considerations

### GPU Memory Optimization

```python
# Gradient checkpointing for large models
if "swin" in backbone_name:
    model.set_grad_checkpointing(True)

# Mixed precision (already in trainer)
scaler = GradScaler()

# Batch size tuning
BATCH_SIZES = {
    "resnet50": 64,
    "efficientnet_b3": 32,  # Larger model
    "mobilenetv3_large": 64,
    "convnext_tiny": 32,
    "swin_tiny": 32,  # Transformer
}
```

### Speed Optimization

```python
# DataLoader optimizations
train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,        # Parallel data loading
    pin_memory=True,      # Faster GPU transfer
    persistent_workers=True,  # Keep workers alive
    prefetch_factor=2,    # Prefetch batches
)
```

---

## Configuration Files

### taskA_baseline.yaml

```yaml
# configs/taskA_baseline.yaml
experiment:
  name: "taskA_baseline"
  task: "taskA"
  num_classes: 4

model:
  backbone: "resnet50"  # Change per model
  pretrained: true
  freeze_backbone: true

data:
  data_root: "data"
  img_size: 224
  batch_size: 64
  num_workers: 4
  augmentation:
    horizontal_flip: 0.5
    vertical_flip: 0.5
    rotation_degrees: 30
    color_jitter: true

training:
  epochs: 50
  lr: 0.0001
  weight_decay: 0.0001
  scheduler: "cosine"
  T_max: 50
  label_smoothing: 0.1
  use_amp: true

evaluation:
  metrics:
    - accuracy
    - precision
    - recall
    - f1_macro
    - confusion_matrix

profiling:
  num_runs: 100
  warmup_runs: 10
```

### taskA_90rotation.yaml

```yaml
# configs/taskA_90rotation.yaml
# Same as baseline but with 90° rotation augmentation
augmentation:
  horizontal_flip: 0.5
  vertical_flip: 0.5
  rotation_90: true  # Key difference
  rotation_degrees: 0  # Not used when rotation_90 is true
  color_jitter: true
```
