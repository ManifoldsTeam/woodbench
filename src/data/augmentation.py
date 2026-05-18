"""Module augmentation pipeline cho minority classes trong Task A.

Cung cap AugmentationMethod enum va AugmentationPipeline class de sinh anh
augmented tu anh goc bang cac phep bien doi hinh anh. Su dung albumentations
lam backend cho hieu suat cao tren anh lon (2560x2048).

Anh mat cat ngang go la isotropic (khong co huong tu nhien), cho phep ap dung
rotation/flip ma khong mat thong tin ngu nghia.
"""

from enum import Enum
from typing import List, Tuple

import albumentations as A
import numpy as np


class AugmentationMethod(Enum):
    """Cac phep bien doi hinh anh ho tro cho augmentation.

    Bao gom 7 methods: flip, rotation, crop+resize, color jitter, transpose.
    Rotation(90) khong nam trong danh sach nay (danh rieng cho ablation study).
    """

    HFLIP = "hflip"
    VFLIP = "vflip"
    ROT180 = "rot180"
    ROT270 = "rot270"
    CROP_RESIZE = "crop_resize"
    COLOR_JITTER = "color_jitter"
    TRANSPOSE = "transpose"


# Kich thuoc goc cua anh trong dataset (width x height)
ORIGINAL_WIDTH = 2560
ORIGINAL_HEIGHT = 2048

# Tham so mac dinh cho ColorJitter
DEFAULT_COLOR_PARAMS = {
    "brightness": 0.15,
    "contrast": 0.15,
    "saturation": 0.1,
    "hue": 0.05,
}


class AugmentationPipeline:
    """Pipeline augmentation cho minority classes.

    Thiet ke stateless: nhan config va tra ve anh augmented.
    Su dung albumentations lam backend de xu ly anh lon hieu qua.

    Args:
        seed: Random seed cho reproducibility (mac dinh 42)
        target_count: Tuple (min, max) so luong mau muc tieu sau augmentation
        crop_min_area: Ti le dien tich toi thieu khi crop (mac dinh 0.7)
        color_params: Dict cau hinh ColorJitter (brightness, contrast, saturation, hue)
    """

    def __init__(
        self,
        seed: int = 42,
        target_count: Tuple[int, int] = (1500, 1700),
        crop_min_area: float = 0.7,
        color_params: dict = None,
    ):
        self.seed = seed
        self.target_count = target_count
        self.crop_min_area = crop_min_area
        self.color_params = color_params if color_params is not None else DEFAULT_COLOR_PARAMS.copy()

        # Khoi tao random state de dam bao reproducibility
        self._rng = np.random.default_rng(seed)

    def get_transform(self, method: AugmentationMethod) -> A.Compose:
        """Tra ve albumentations transform tuong ung voi method cu the.

        Moi transform duoc wrap trong A.Compose voi p=1.0 de dam bao
        luon duoc ap dung khi goi.

        Args:
            method: Phep bien doi can lay transform

        Returns:
            A.Compose chua transform tuong ung

        Raises:
            ValueError: Neu method khong hop le
        """
        if method == AugmentationMethod.HFLIP:
            transform = A.HorizontalFlip(p=1.0)

        elif method == AugmentationMethod.VFLIP:
            transform = A.VerticalFlip(p=1.0)

        elif method == AugmentationMethod.ROT180:
            transform = A.Rotate(limit=(180, 180), p=1.0, border_mode=0)

        elif method == AugmentationMethod.ROT270:
            transform = A.Rotate(limit=(270, 270), p=1.0, border_mode=0)

        elif method == AugmentationMethod.CROP_RESIZE:
            # Crop it nhat 70% dien tich goc, resize ve kich thuoc ban dau
            transform = A.RandomResizedCrop(
                size=(ORIGINAL_HEIGHT, ORIGINAL_WIDTH),
                scale=(self.crop_min_area, 0.9),
                p=1.0,
            )

        elif method == AugmentationMethod.COLOR_JITTER:
            # Tham so vua phai de giu wood grain visibility
            transform = A.ColorJitter(
                brightness=self.color_params["brightness"],
                contrast=self.color_params["contrast"],
                saturation=self.color_params["saturation"],
                hue=self.color_params["hue"],
                p=1.0,
            )

        elif method == AugmentationMethod.TRANSPOSE:
            # Transpose hoan doi H va W, can resize lai ve kich thuoc goc
            return A.Compose(
                [
                    A.Transpose(p=1.0),
                    A.Resize(height=ORIGINAL_HEIGHT, width=ORIGINAL_WIDTH, p=1.0),
                ],
                seed=self.seed,
            )

        else:
            raise ValueError(f"Method khong hop le: {method}")

        return A.Compose([transform], seed=self.seed)

    def augment_image(self, image: np.ndarray, method: AugmentationMethod) -> np.ndarray:
        """Ap dung 1 augmentation method len 1 anh.

        Args:
            image: Numpy array (H, W, C) chua anh goc, dtype uint8
            method: Phep bien doi can ap dung

        Returns:
            Numpy array (H, W, C) chua anh da augment, cung dtype voi input
        """
        transform = self.get_transform(method)
        result = transform(image=image)
        return result["image"]

    def compute_augmentation_plan(
        self, original_count: int, target_count: int
    ) -> List[Tuple[int, AugmentationMethod]]:
        """Tinh toan plan augmentation: anh nao duoc augment bang method nao.

        Phan phoi deu cac method across images theo round-robin.
        Moi anh goc se duoc augment nhieu lan voi cac method khac nhau.

        Args:
            original_count: So luong anh goc hien co
            target_count: So luong mau muc tieu (bao gom ca originals)

        Returns:
            List of (image_index, method) tuples.
            image_index la chi so cua anh goc (0-based, modulo original_count).
            method la AugmentationMethod se ap dung.
        """
        # So luong anh can sinh them
        needed = target_count - original_count
        if needed <= 0:
            return []

        # Danh sach tat ca methods de round-robin
        all_methods = list(AugmentationMethod)

        plan = []
        for i in range(needed):
            # Round-robin qua cac methods
            method = all_methods[i % len(all_methods)]
            # Round-robin qua cac anh goc
            image_index = i % original_count
            plan.append((image_index, method))

        return plan
