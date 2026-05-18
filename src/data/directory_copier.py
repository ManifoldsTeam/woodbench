"""Module copy thu muc du lieu tu data/split/ sang data/task-A/.

Chuyen doi symlinks thanh real files, dam bao output khong chua symbolic links.
"""

import shutil
from pathlib import Path
from typing import List

# Cac dinh dang anh duoc ho tro
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


class DirectoryCopier:
    """Copy files tu data/split/ sang data/task-A/ dang real files.

    Class nay dam bao:
    - Tao cau truc thu muc data/task-A/{split}/{family}/{species}/
    - Copy tat ca file anh duoi dang real files (khong symlink)
    - Resolve symlinks khi copy (source data dung symlinks toi data/raw/)
    """

    def copy_split(
        self,
        src_root: Path,
        dst_root: Path,
        split: str,
        families: List[str],
    ) -> int:
        """Copy toan bo anh cua 1 split cho cac families chi dinh.

        Tao cau truc thu muc va copy file thuc (resolve symlinks).
        Su dung shutil.copy2 de giu nguyen metadata va tu dong resolve symlinks.

        Args:
            src_root: Thu muc goc chua du lieu (VD: data/split/)
            dst_root: Thu muc dich (VD: data/task-A/)
            split: Ten split can copy (train, val, test)
            families: Danh sach cac family can copy

        Returns:
            So luong file da copy thanh cong
        """
        copied_count = 0
        src_split_dir = Path(src_root) / split

        if not src_split_dir.exists():
            return copied_count

        for family in families:
            src_family_dir = src_split_dir / family

            if not src_family_dir.exists():
                continue

            # Duyet qua cac thu muc species trong family
            for species_dir in src_family_dir.iterdir():
                if not species_dir.is_dir():
                    continue

                # Tao thu muc dich: data/task-A/{split}/{family}/{species}/
                dst_species_dir = Path(dst_root) / split / family / species_dir.name
                dst_species_dir.mkdir(parents=True, exist_ok=True)

                # Copy tung file anh trong thu muc species
                for src_file in species_dir.iterdir():
                    if not src_file.is_file():
                        continue

                    # Chi copy file co dinh dang anh hop le
                    if src_file.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue

                    dst_file = dst_species_dir / src_file.name

                    # shutil.copy2 tu dong resolve symlinks va giu metadata
                    shutil.copy2(src_file, dst_file)
                    copied_count += 1

        return copied_count
