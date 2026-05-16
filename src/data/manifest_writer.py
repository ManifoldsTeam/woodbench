"""Module ghi manifest CSV theo doi provenance cua qua trinh augmentation.

Manifest CSV luu tru thong tin ve moi file augmented: duong dan goc,
duong dan augmented, phuong phap augmentation, ho va loai thuc vat.
"""

import csv
from pathlib import Path
from typing import List, Dict


class ManifestWriter:
    """Ghi manifest CSV theo doi provenance cua cac file augmented.

    Class nay quan ly danh sach cac entry augmentation va ghi ra file CSV
    voi cac cot: original_path, augmented_path, method, family, species.

    Args:
        output_path: Duong dan file CSV dau ra (Path object).
    """

    # Cac cot trong file CSV manifest
    COLUMNS = ["original_path", "augmented_path", "method", "family", "species"]

    def __init__(self, output_path: Path) -> None:
        """Khoi tao ManifestWriter voi duong dan file CSV dau ra.

        Args:
            output_path: Duong dan noi file CSV se duoc luu.
        """
        self.output_path = Path(output_path)
        self._entries: List[Dict[str, str]] = []

    def add_entry(
        self,
        original_path: str,
        augmented_path: str,
        method: str,
        family: str,
        species: str,
    ) -> None:
        """Them mot entry vao danh sach manifest.

        Moi entry tuong ung voi mot file augmented duoc sinh ra tu file goc.

        Args:
            original_path: Duong dan tuong doi den file anh goc.
            augmented_path: Duong dan tuong doi den file anh augmented.
            method: Ten phuong phap augmentation da ap dung.
            family: Ten ho thuc vat (family).
            species: Ten loai thuc vat (species).
        """
        entry = {
            "original_path": original_path,
            "augmented_path": augmented_path,
            "method": method,
            "family": family,
            "species": species,
        }
        self._entries.append(entry)

    def save(self) -> Path:
        """Ghi tat ca entries ra file CSV.

        Tao thu muc cha neu chua ton tai. File CSV se co header row
        va moi entry la mot dong.

        Returns:
            Duong dan file CSV da duoc ghi.
        """
        # Tao thu muc cha neu chua ton tai
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writeheader()
            writer.writerows(self._entries)

        return self.output_path
