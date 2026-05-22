"""Dataset module for IC4SD-Wood-Conifer20v1.

Structure: data/{split}/{family}/{species}/images.jpg
Supports both Task A (family-level) and Task B (species-level) classification.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset


class ConiferDataset(Dataset):
    """Dataset for Conifer wood classification.

    Structure: data/{split}/{family}/{species}/images.jpg

    Args:
        data_root: Root directory containing train/val/test folders
        split: 'train' | 'val' | 'test'
        transform: Optional transform to apply
        task: 'taskA' (family) or 'taskB' (species)
    """

    FAMILY_CLASSES = [
        "Cupressaceae",
        "Pinaceae",
        "Podocarpaceae",
        "Taxaceae",
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

        if task == "taskA":
            self.classes = self.FAMILY_CLASSES
        else:
            self.classes = self._get_species_classes()

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = self._collect_samples()

    def _get_species_classes(self) -> List[str]:
        """Infer species classes from the dataset directory structure.

        Structure: data/{split}/{family}/{species}/images.jpg
        """
        species = set()
        split_path = self.data_root / self.split
        if not split_path.exists():
            return []

        for family_folder in split_path.iterdir():
            if family_folder.is_dir():
                for species_folder in family_folder.iterdir():
                    if species_folder.is_dir():
                        species.add(species_folder.name)

        return sorted(species)

    def _collect_samples(self) -> List[Tuple[Path, int]]:
        """Walk directory tree and collect (image_path, label) tuples.

        Structure: data/{split}/{family}/{species}/images.jpg
        """
        samples = []
        split_path = self.data_root / self.split

        if not split_path.exists():
            return samples

        for family_folder in split_path.iterdir():
            if not family_folder.is_dir():
                continue

            if self.task == "taskA":
                if family_folder.name not in self.class_to_idx:
                    continue
                label = self.class_to_idx[family_folder.name]
                for species_folder in family_folder.iterdir():
                    if species_folder.is_dir():
                        for img_path in species_folder.glob("*.jpg"):
                            samples.append((img_path, label))
                        for img_path in species_folder.glob("*.jpeg"):
                            samples.append((img_path, label))
                        for img_path in species_folder.glob("*.png"):
                            samples.append((img_path, label))
            else:
                for species_folder in family_folder.iterdir():
                    if species_folder.is_dir() and species_folder.name in self.class_to_idx:
                        label = self.class_to_idx[species_folder.name]
                        for img_path in species_folder.glob("*.jpg"):
                            samples.append((img_path, label))
                        for img_path in species_folder.glob("*.jpeg"):
                            samples.append((img_path, label))
                        for img_path in species_folder.glob("*.png"):
                            samples.append((img_path, label))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_distribution(self) -> Dict[str, int]:
        """Get the distribution of samples per class."""
        distribution = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            distribution[self.classes[label]] += 1
        return distribution


class FlatConiferDataset(Dataset):
    """Flat dataset: data/{split}/{species}/images.jpg (no family folder).

    Used for HuggingFace datasets (e.g. Task B) with structure:
        train/
            Chamaecyparis_obtusa/
            Cryptomeria_japonica/
            ...
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        task: str = "taskB",
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.task = task

        split_path = self.data_root / split
        if not split_path.exists():
            self.classes = []
            self.class_to_idx = {}
            self.samples = []
            return

        # Classes = subfolders directly under split/
        self.classes = sorted([
            d.name for d in split_path.iterdir() if d.is_dir()
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.samples = self._collect_samples()

    def _collect_samples(self) -> List[Tuple[Path, int]]:
        samples = []
        split_path = self.data_root / self.split
        for cls_folder in split_path.iterdir():
            if not cls_folder.is_dir():
                continue
            if cls_folder.name not in self.class_to_idx:
                continue
            label = self.class_to_idx[cls_folder.name]
            for img_path in cls_folder.glob("*.jpg"):
                samples.append((img_path, label))
            for img_path in cls_folder.glob("*.jpeg"):
                samples.append((img_path, label))
            for img_path in cls_folder.glob("*.png"):
                samples.append((img_path, label))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_class_distribution(self) -> Dict[str, int]:
        distribution = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            distribution[self.classes[label]] += 1
        return distribution


def _detect_structure(data_root: Path, split: str) -> str:
    """Detect whether dataset has nested (family/species) or flat (species) structure."""
    split_path = data_root / split
    if not split_path.exists():
        return "flat"
    subdirs = [d for d in split_path.iterdir() if d.is_dir()]
    if not subdirs:
        return "flat"
    # Check if subdirs have further subdirs (nested) or not (flat)
    first = subdirs[0]
    has_nested = any(d.is_dir() for d in first.iterdir())
    return "nested" if has_nested else "flat"


def get_dataloaders(
    data_root: str | Path,
    batch_size: int = 64,
    num_workers: int = 4,
    use_90rotation: bool = False,
    img_size: int = 224,
    task: str = "taskA",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders with appropriate transforms.

    Args:
        data_root: Path to dataset root (data/split/ or data/task-B/)
        batch_size: Batch size
        num_workers: DataLoader workers
        use_90rotation: Whether to use 90-degree rotation augmentation
        img_size: Input image size
        task: 'taskA' (family) or 'taskB' (species)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from .transforms import get_train_transforms, get_val_transforms

    data_root = Path(data_root)

    # Detect structure for taskB (nested vs flat)
    use_flat = task == "taskB" and _detect_structure(data_root, "train") == "flat"
    DatasetCls = FlatConiferDataset if use_flat else ConiferDataset

    train_transform = get_train_transforms(img_size=img_size, use_90rotation=use_90rotation)
    val_transform = get_val_transforms(img_size=img_size)

    train_dataset = DatasetCls(data_root, split="train", transform=train_transform, task=task)
    val_dataset = DatasetCls(data_root, split="val", transform=val_transform, task=task)
    test_dataset = DatasetCls(data_root, split="test", transform=val_transform, task=task)

    if len(train_dataset) == 0:
        raise ValueError(
            f"[get_dataloaders] No training samples found in {data_root}/train. "
            f"task={task}, structure={_detect_structure(data_root, 'train')}. "
            f"Subdirs: {list((data_root / 'train').iterdir()) if (data_root / 'train').exists() else 'N/A'}"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return train_loader, val_loader, test_loader
