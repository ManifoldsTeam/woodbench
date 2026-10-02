"""Reuse the origin identities materialized for the final Phase 1 runs.

The investigator notebook defines class::stem keys, shuffles sorted origins
inside each species with numpy.default_rng(seed), and copies whole origins.
CV retains those identities and distributes them over folds instead of the
original 70/15/15 split. The existing dataset folders are never rewritten.
"""

import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[4]
PHASE1_MANIFEST = (
    ROOT
    / "docs/reports/phase_1/phase_1_leakage_origin_investigation"
    / "task-B-clean_group_split_plan.csv"
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
# The manifest predates the final Phase 1 exclusion of these three classes.
PHASE1_EXCLUDED_CLASSES = {
    "Calocedrus sp",
    "Cupressus funebris (Cupressus pendula)",
    "Pinus sp",
}


def infer_origin_key(path: Path) -> str:
    """Exact key convention from the Phase 1 investigator notebook, cell 6."""
    stem = re.sub(r"\s*\(\d+\)$", "", path.stem)
    return f"{path.parent.name}::{stem}"


def load_origin_samples(data_root: Path, manifest_path: Path, splits: tuple) -> list:
    """Resolve original keys through the Phase 1 copy/rename plan, cell 15.

    Replaying that plan also handles __from-<split>__<md5-prefix> filenames.
    Keys are never inferred from renamed destinations or missing metadata.
    """
    data_root, manifest_path = Path(data_root), Path(manifest_path)
    with manifest_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "new_split",
            "old_split",
            "class_name",
            "filename",
            "origin_key",
            "md5",
        }
        if not required <= set(reader.fieldnames or []):
            raise ValueError(
                f"Phase 1 manifest missing columns: {sorted(required - set(reader.fieldnames or []))}"
            )
        rows = list(reader)

    materialized = {}
    for row in rows:
        original = Path(row["class_name"]) / row["filename"]
        if row["origin_key"] != infer_origin_key(original):
            raise ValueError(
                f"Origin differs from Phase 1 manifest convention: {original}"
            )
        relative = Path(row["new_split"]) / row["class_name"] / row["filename"]
        if relative.as_posix() in materialized:
            filename = Path(row["filename"])
            old_split = row["old_split"].replace("/", "-")
            relative = relative.with_name(
                f"{filename.stem}__from-{old_split}__{row['md5'][:8]}{filename.suffix}"
            )
        if relative.as_posix() in materialized:
            raise ValueError(f"Ambiguous destination in Phase 1 manifest: {relative}")
        materialized[relative.as_posix()] = row

    records = []
    for split in splits:
        split_dir = data_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing original dataset split: {split_dir}")
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for path in sorted(class_dir.iterdir()):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                relative = path.relative_to(data_root).as_posix()
                if relative not in materialized:
                    raise ValueError(f"Image not found in Phase 1 manifest: {relative}")
                row = materialized[relative]
                records.append(
                    {
                        "path": str(path),
                        "species": class_dir.name,
                        "origin_key": row["origin_key"],
                        "md5": row["md5"],
                        "source_split": split,
                        "source_filename": row["filename"],
                    }
                )

    expected = {
        name
        for name, row in materialized.items()
        if row["new_split"] in splits
        and row["class_name"] not in PHASE1_EXCLUDED_CLASSES
    }
    present = {Path(row["path"]).relative_to(data_root).as_posix() for row in records}
    if expected != present:
        raise ValueError(
            f"Images missing from Phase 1 manifest dataset: {sorted(expected - present)[:5]}"
        )
    if not records:
        raise ValueError("No images matched the Phase 1 manifest")

    origin_splits = defaultdict(set)
    hash_splits = defaultdict(set)
    for row in records:
        origin_splits[row["origin_key"]].add(row["source_split"])
        hash_splits[row["md5"]].add(row["source_split"])
    if any(len(s) > 1 for s in origin_splits.values()) or any(
        len(s) > 1 for s in hash_splits.values()
    ):
        raise ValueError("Phase 1 origin or MD5 occurs in more than one source split")
    return records


def make_origin_folds(records: list, n_folds: int = 5, seed: int = 42) -> list:
    """Shuffle origins separately per species, as in Phase 1, then partition."""
    if n_folds < 2:
        raise ValueError("CV requires at least two folds")
    rng = np.random.default_rng(seed)
    held_out = [set() for _ in range(n_folds)]
    species = sorted({row["species"] for row in records})
    for name in species:
        origins = np.array(
            sorted({row["origin_key"] for row in records if row["species"] == name})
        )
        if len(origins) < n_folds:
            raise ValueError(
                f"{name} has {len(origins)} origins, fewer than {n_folds} folds"
            )
        rng.shuffle(origins)
        for fold, groups in enumerate(np.array_split(origins, n_folds)):
            held_out[fold].update(groups)

    keys = np.array([row["origin_key"] for row in records])
    splits = []
    for fold, val_origins in enumerate(held_out):
        val_mask = np.isin(keys, list(val_origins))
        train_idx, val_idx = np.flatnonzero(~val_mask), np.flatnonzero(val_mask)
        train_origins = set(keys[train_idx])
        if train_origins & val_origins:
            raise ValueError("Origin leakage across CV fold")
        train_hashes = {records[i]["md5"] for i in train_idx}
        val_hashes = {records[i]["md5"] for i in val_idx}
        if train_hashes & val_hashes:
            raise ValueError("MD5 leakage across CV fold")
        splits.append(
            {
                "fold": fold,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "train_origins": train_origins,
                "val_origins": val_origins,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
            }
        )
    return splits


class OriginKFoldDataset(Dataset):
    """One fixed class mapping and Phase 1 keys across all original splits."""

    def __init__(
        self,
        data_root: Path,
        split: str = "all",
        transform=None,
        classes: list = None,
        manifest_path: Path = PHASE1_MANIFEST,
    ):
        self.data_root, self.split, self.transform = Path(data_root), split, transform
        source_splits = ("train", "val", "test") if split == "all" else (split,)
        self.records = load_origin_samples(self.data_root, manifest_path, source_splits)
        discovered = sorted({row["species"] for row in self.records})
        self.classes = discovered if classes is None else classes
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        self.samples = [
            (row["path"], self.class_to_idx[row["species"]], row["origin_key"])
            for row in self.records
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, origin_key = self.samples[idx]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, origin_key

    def get_splits_by_origin(self, n_folds: int = 5, seed: int = 42):
        return make_origin_folds(self.records, n_folds, seed)


class SubsetWithTransform(Dataset):
    """Apply the selected train/eval transform to the original image once."""

    def __init__(self, dataset, indices, transform=None):
        self.dataset, self.indices = dataset, indices
        self.transform = transform if transform is not None else dataset.transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path, label, _ = self.dataset.samples[self.indices[idx]]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
