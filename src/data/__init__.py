"""Data module for Conifer Family Classification."""

from .augmentation import AugmentationPipeline, AugmentationMethod
from .dataset import ConiferDataset, FlatConiferDataset, get_dataloaders
from .directory_copier import DirectoryCopier
from .manifest_writer import ManifestWriter
from .splitter import create_stratified_split, verify_split
from .transforms import get_train_transforms, get_val_transforms

__all__ = [
    # Augmentation (Phase 0)
    "AugmentationPipeline",
    "AugmentationMethod",
    "DirectoryCopier",
    "ManifestWriter",
    # Dataset & transforms
    "ConiferDataset",
    "get_dataloaders",
    "get_train_transforms",
    "get_val_transforms",
    # Splitter
    "create_stratified_split",
    "verify_split",
]
