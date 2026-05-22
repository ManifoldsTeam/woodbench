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

    train_transform = get_train_transforms(img_size=img_size, use_90rotation=use_90rotation)
    val_transform = get_val_transforms(img_size=img_size)

    train_dataset = ConiferDataset(data_root, split="train", transform=train_transform, task=task)
    val_dataset = ConiferDataset(data_root, split="val", transform=val_transform, task=task)
    test_dataset = ConiferDataset(data_root, split="test", transform=val_transform, task=task)

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
