"""Stratified data splitting for IC4SD-Wood-Conifer20v1.

Creates stratified train/val/test splits at the species level,
preserving the 2-level folder structure: data/{split}/{family}/{species}/.
"""

from dataclasses import dataclass

from pathlib import Path
from typing import Dict, List

import numpy as np
import shutil


@dataclass
class SplitConfig:
    """Configuration for stratified split ratios."""
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15
    seed: int = 42

    def __post_init__(self):
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")


def create_stratified_split(
    data_root: Path,
    output_root: Path,
    config: SplitConfig,
) -> Dict[str, Dict[str, int]]:
    """Create stratified train/val/test split at species level.

    Splits images within each species proportionally across train/val/test.
    Creates folder structure: output/{train,val,test}/{family}/{species}/

    Returns:
        Dictionary with per-split, per-family sample counts
    """
    rng = np.random.default_rng(config.seed)
    species_paths = _discover_species(data_root)
    _create_split_folders(output_root, species_paths)

    split_counts = {s: {} for s in ["train", "val", "test"]}

    for species_name, species_info in species_paths.items():
        family = species_info["family"]
        images = _load_images(species_info["path"])
        if not images:
            continue
        rng.shuffle(images)
        n = len(images)
        n_train = int(n * config.train)
        n_val = int(n * config.val)

        for split_name, split_images in {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }.items():
            _copy_images(split_images, output_root / split_name / family / species_name)
            for s in ["train", "val", "test"]:
                split_counts[s].setdefault(family, 0)
            split_counts[split_name][family] += len(split_images)

    return split_counts


def _discover_species(data_root: Path) -> Dict:
    """Find all species folders in the dataset."""
    species_paths = {}
    for family_folder in sorted(data_root.iterdir()):
        if not family_folder.is_dir():
            continue
        for species_folder in sorted(family_folder.iterdir()):
            if species_folder.is_dir():
                species_paths[species_folder.name] = {
                    "family": family_folder.name,
                    "path": species_folder,
                }
    return species_paths


def _create_split_folders(output_root: Path, species_paths: Dict) -> None:
    """Create split directory structure."""
    by_family: Dict[str, List[str]] = {}
    for species_name, info in species_paths.items():
        by_family.setdefault(info["family"], []).append(species_name)
    for split in ["train", "val", "test"]:
        for family, species_list in by_family.items():
            for species_name in species_list:
                (output_root / split / family / species_name).mkdir(parents=True, exist_ok=True)


def _load_images(source: Path) -> list:
    """Load all image paths from a source directory."""
    return sorted([
        f for f in source.iterdir()
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])


def _copy_images(images: list, dest_dir: Path) -> None:
    """Copy images to destination directory."""
    for img_path in images:
        shutil.copy2(img_path, dest_dir / img_path.name)


def verify_split(
    data_root: Path,
    expected_splits: List[str] = ["train", "val", "test"],
) -> Dict[str, Dict]:
    """Verify split integrity and return statistics.

    Args:
        data_root: Root of split dataset
        expected_splits: List of expected splits

    Returns:
        Dictionary with per-split statistics
    """
    stats = {}
    total = 0

    for split in expected_splits:
        split_path = data_root / split
        if not split_path.exists():
            stats[split] = {"exists": False, "total": 0, "families": {}}
            continue

        family_counts = {}
        split_total = 0

        for family_folder in sorted(split_path.iterdir()):
            if not family_folder.is_dir():
                continue
            species_count = 0
            for species_folder in family_folder.iterdir():
                if species_folder.is_dir():
                    n_images = len([
                        f for f in species_folder.iterdir()
                        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
                    ])
                    species_count += n_images
                    split_total += n_images
            family_counts[family_folder.name] = species_count

        stats[split] = {
            "exists": True,
            "total": split_total,
            "families": family_counts,
        }
        total += split_total

    stats["_total"] = total
    return stats
