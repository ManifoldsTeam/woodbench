#!/usr/bin/env python3
"""
Hiển thị các ảnh cần rà soát theo cấu trúc:

    split -> phân loài -> các ảnh được đánh dấu

Cách chạy:
    python show_review_images.py /duong/dan/toi/dataset

Ví dụ:
    python show_review_images.py ./dataset --cols 4 --page-size 16
    python show_review_images.py ./dataset --save-dir review_sheets --no-show

Mặc định, những phân loài chỉ được ghi tên nhưng không có file/range cụ thể
sẽ không tải toàn bộ ảnh. Dùng --include-class-only để hiển thị toàn bộ ảnh
của các phân loài đó.
"""

from __future__ import annotations

import argparse
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from PIL import Image, ImageOps, UnidentifiedImageError


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class FileTarget:
    filename: str
    note: str = ""


@dataclass(frozen=True)
class RangeTarget:
    start: int
    end: int
    note: str = ""


# Những phân loài chỉ có tên trong danh sách ban đầu.
# Chúng chỉ được hiển thị khi truyền --include-class-only.
CLASS_ONLY: dict[str, list[str]] = {
    "test": [
        "Araucaria araucana",
        "Calocedrus rupestris",
        "Picea glauca",
        "Pinus cernua",
        "Pinus dalatensis",
        "Pinus krempfii",
        "Pinus radiata",
        "Pseudotsuga brevifolia",
        "Taxus wallichiana",
        "Thujopsis dolabrata",
        "Xanthocyparis vietnamensis",
    ],
    "train": [
        "Araucaria araucana",
        "Cryptomeria japonica",
        "Cunninghamia lanceolata",
        "Chamaecyparis obtusa",
        "Picea glauca",
        "Pinus cernua",
        "Pinus radiata",
        "Podocarpus pilgeri",
        "Pseudotsuga brevifolia",
        "Taxus wallichiana",
        "Thujopsis dolabrata",
        "Xanthocyparis vietnamensis",
    ],
    "val": [
        "Araucaria araucana",
        "Calocedrus rupestris",
        "Chamaecyparis obtusa",
        "Cryptomeria japonica",
        "Fokienia hodginsii",
        "Picea glauca",
        "Pinus cernua",
        "Pinus dalatensis",
        "Pinus radiata",
        "Pseudotsuga brevifolia",
        "Taxus wallichiana",
        "Thujopsis dolabrata",
        "Xanthocyparis vietnamensis",
    ],
}


TARGETS: dict[str, dict[str, list[FileTarget | RangeTarget]]] = {
    "test": {
        "Callitris columellaris": [
            FileTarget("14.png"),
            FileTarget("127.png"),
            FileTarget("127__from-val__06c2f2ac.png"),
            FileTarget("132.png"),
            FileTarget("138.png"),
            FileTarget("162.png"),
            FileTarget("165.png"),
            FileTarget("193.png"),
        ],
        "Cryptomeria japonica": [
            FileTarget("3350. Cryptomeria japonica.4.(115).png"),
            FileTarget("3350. Cryptomeria japonica.4.(116).png"),
            FileTarget("3350. Cryptomeria japonica.4.(119).png"),
            FileTarget("3350. Cryptomeria japonica.4.(123).png"),
            FileTarget("3350. Cryptomeria japonica.4.(130).png", "Hơi cháy"),
            FileTarget("3350. Cryptomeria japonica.4.(131).png", "Hơi cháy"),
            FileTarget("3350. Cryptomeria japonica.4.(132).png", "Hơi cháy"),
            FileTarget("3350. Cryptomeria japonica.4.(133).png", "Hơi cháy"),
        ],
        "Cunninghamia lanceolata": [
            FileTarget("16__from-val__98d74a4d.png", "Nghi ngờ"),
            FileTarget("113__from-val__091ca36d.png", "Nghi ngờ"),
        ],
        "Chamaecyparis obtusa": [
            FileTarget(
                "3364. Chamaecyparis obtusa.5(132).png",
                "Cháy một ít ở viền phải",
            ),
        ],
        "Fokienia hodginsii": [
            FileTarget("pơ mu97.png", "Nghi ngờ"),
            FileTarget("pơmu72.png", "Mất một góc tam giác bên trên trái"),
            FileTarget("pơmu79.png", "Cháy một ít góc dưới phải"),
        ],
        "Podocarpus pilgeri": [
            FileTarget(
                "3386. Podocarpus pilgeri.1 (10).png",
                "Nghi ngờ cháy dưới trái",
            ),
        ],
    },
    "train": {
        "Callitris columellaris": [
            FileTarget("10.png", "Cháy sáng ở một vài đốm"),
            FileTarget("17__from-test__996e2bc5.png", "Nghi ngờ cháy dưới phải"),
            FileTarget("18.png", "Cháy sáng"),
            FileTarget("125.png", "Nghi ngờ cháy sáng"),
            FileTarget("143.png", "Nghi ngờ cháy sáng trên phải"),
            FileTarget("160__from-test__97905399.png", "Nghi ngờ"),
            FileTarget("184.png", "Nghi ngờ"),
            FileTarget("185__from-test__78bcae18.png"),
        ],
        "Calocedrus rupestris": [
            RangeTarget(
                20,
                33,
                "Hình chữ nhật đen; ngày nhập nhỏ hơn",
            ),
            RangeTarget(
                288,
                292,
                "Hình chữ nhật đen; ngày nhập nhỏ hơn",
            ),
        ],
        "Chamaecyparis obtusa": [
            FileTarget("3350. Cryptomeria japonica.1.(13).png", "Nghi ngờ"),
            FileTarget("3350. Cryptomeria japonica.6.(194).png", "Nghi ngờ"),
            FileTarget("3350. Cryptomeria japonica.7.(223).png"),
        ],
        "Fokienia hodginsii": [
            FileTarget("pơ mu19.png", "Nghi ngờ"),
            FileTarget("pơ mu71.png", "Nghi ngờ"),
            FileTarget("pơ mu35.png", "Nghi ngờ"),
            FileTarget("pơmu44__from-val__e58d53f5.png", "Nghi ngờ"),
            FileTarget("pơmu48__from-val__2f4a7924.png", "Nghi ngờ"),
            FileTarget("pơ mu63.png", "Nghi ngờ"),
            FileTarget("pơ mu95.png", "Nghi ngờ"),
        ],
        "Pinus dalatensis": [
            FileTarget("3393. Pinus dalatensis.1 (1).png"),
            FileTarget("3393. Pinus dalatensis.1 (6).png"),
            FileTarget("3393. Pinus dalatensis.1 (8).png"),
            FileTarget("3393. Pinus dalatensis.1 (9).png"),
            FileTarget("3393. Pinus dalatensis.1 (10).png"),
            FileTarget("3393. Pinus dalatensis.1 (11).png"),
            FileTarget("3393. Pinus dalatensis.1 (27).png"),
            FileTarget("3393. Pinus dalatensis.1 (68).png"),
            FileTarget("3393. Pinus dalatensis.1 (96).png", "Một góc nhỏ bên phải"),
            FileTarget("3393. Pinus dalatensis.1 (115).png", "Trên phải nhỏ"),
        ],
        "Pinus krempfii": [
            FileTarget("3384. Pinus krempfii (60).png"),
            FileTarget(
                "3384. Pinus krempfii.3 (92).png",
                "Cháy một ít bên trên",
            ),
        ],
    },
    "val": {
        "Callitris columellaris": [
            FileTarget("15.png", "Nghi ngờ"),
            FileTarget("126.png"),
            FileTarget("128__from-val__7d4c5343.png"),
            FileTarget("190__from-test__d12db15c.png", "Nghi ngờ"),
            FileTarget("190__from-val__24eaae71.png", "Nghi ngờ"),
        ],
        "Cunninghamia lanceolata": [
            FileTarget("13.png", "Nghi ngờ"),
        ],
        "Podocarpus pilgeri": [
            RangeTarget(221, 222, "Hai ảnh giống hệt nhau"),
        ],
    },
}


def normalize_unicode(text: str) -> str:
    """Chuẩn hóa Unicode để khớp cả pơ và dạng ký tự tổ hợp pơ."""
    return unicodedata.normalize("NFC", text).casefold()


def compact_name(text: str) -> str:
    """
    So khớp dự phòng: bỏ khoảng trắng.

    Hữu ích với các tên như:
        pơ mu97.png
        pơmu97.png
    """
    return re.sub(r"\s+", "", normalize_unicode(text))


def natural_key(text: str) -> list[object]:
    return [
        int(part) if part.isdigit() else normalize_unicode(part)
        for part in re.split(r"(\d+)", text)
    ]


def image_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []

    return sorted(
        (
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: natural_key(path.name),
    )


def resolve_species_dir(split_dir: Path, requested_species: str) -> Path | None:
    direct = split_dir / requested_species
    if direct.is_dir():
        return direct

    requested_norm = normalize_unicode(requested_species)
    matches = [
        path
        for path in split_dir.iterdir()
        if path.is_dir() and normalize_unicode(path.name) == requested_norm
    ]

    if len(matches) == 1:
        return matches[0]

    return None


def build_filename_indexes(files: Iterable[Path]) -> tuple[
    dict[str, list[Path]],
    dict[str, list[Path]],
]:
    unicode_index: dict[str, list[Path]] = {}
    compact_index: dict[str, list[Path]] = {}

    for path in files:
        unicode_index.setdefault(normalize_unicode(path.name), []).append(path)
        compact_index.setdefault(compact_name(path.name), []).append(path)

    return unicode_index, compact_index


def resolve_file(
    requested_name: str,
    unicode_index: dict[str, list[Path]],
    compact_index: dict[str, list[Path]],
) -> Path | None:
    exact_unicode = unicode_index.get(normalize_unicode(requested_name), [])
    if len(exact_unicode) == 1:
        return exact_unicode[0]

    compact_matches = compact_index.get(compact_name(requested_name), [])
    if len(compact_matches) == 1:
        return compact_matches[0]

    if len(exact_unicode) > 1:
        print(f"[CẢNH BÁO] Trùng tên Unicode: {requested_name}")
    elif len(compact_matches) > 1:
        print(f"[CẢNH BÁO] Tên rút gọn không duy nhất: {requested_name}")

    return None


def final_number(path: Path) -> int | None:
    """
    Lấy số cuối cùng trong tên file.

    Ví dụ:
        3342. Calocedrus rupestris.7.(288).png -> 288
        20.png                                  -> 20
    """
    numbers = re.findall(r"\d+", path.stem)
    return int(numbers[-1]) if numbers else None


def collect_group(
    species_dir: Path,
    targets: list[FileTarget | RangeTarget],
) -> tuple[list[tuple[Path, str]], list[str]]:
    files = image_files(species_dir)
    unicode_index, compact_index = build_filename_indexes(files)

    selected: list[tuple[Path, str]] = []
    missing: list[str] = []
    seen: set[Path] = set()

    for target in targets:
        if isinstance(target, FileTarget):
            resolved = resolve_file(
                target.filename,
                unicode_index=unicode_index,
                compact_index=compact_index,
            )
            if resolved is None:
                missing.append(target.filename)
                continue

            if resolved not in seen:
                selected.append((resolved, target.note))
                seen.add(resolved)
            continue

        range_matches = [
            path
            for path in files
            if (number := final_number(path)) is not None
            and target.start <= number <= target.end
        ]

        if not range_matches:
            missing.append(f"range {target.start}–{target.end}")
            continue

        for path in range_matches:
            if path not in seen:
                selected.append((path, target.note))
                seen.add(path)

    selected.sort(key=lambda item: natural_key(item[0].name))
    return selected, missing


def safe_filename(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'[<>:"/\\|?*]+', "_", text)
    return re.sub(r"\s+", "_", text).strip("._")


def load_for_display(path: Path, thumb_size: int) -> Image.Image:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((thumb_size, thumb_size))
        return image.copy()


def show_group(
    split: str,
    species: str,
    items: list[tuple[Path, str]],
    *,
    cols: int,
    page_size: int,
    thumb_size: int,
    save_dir: Path | None,
    show: bool,
) -> None:
    for page_start in range(0, len(items), page_size):
        page = items[page_start : page_start + page_size]
        page_number = page_start // page_size + 1
        total_pages = math.ceil(len(items) / page_size)

        rows = math.ceil(len(page) / cols)
        figure_width = max(10, cols * 3.6)
        figure_height = max(4, rows * 3.8)

        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(figure_width, figure_height),
            squeeze=False,
        )

        fig.suptitle(
            f"{split.upper()} / {species} — "
            f"{len(items)} ảnh — trang {page_number}/{total_pages}",
            fontsize=15,
        )

        flat_axes = axes.ravel()

        for axis, (path, note) in zip(flat_axes, page):
            axis.axis("off")

            try:
                image = load_for_display(path, thumb_size)
                axis.imshow(image)
            except (OSError, UnidentifiedImageError) as error:
                axis.text(
                    0.5,
                    0.5,
                    f"Không đọc được ảnh\n{error}",
                    ha="center",
                    va="center",
                    wrap=True,
                )

            title = path.name
            if note:
                title += f"\n⚠ {note}"

            axis.set_title(title, fontsize=8, wrap=True)

        for axis in flat_axes[len(page) :]:
            axis.axis("off")

        fig.tight_layout(rect=(0, 0, 1, 0.95))

        if save_dir is not None:
            output_dir = save_dir / split
            output_dir.mkdir(parents=True, exist_ok=True)
            output_name = (
                f"{safe_filename(species)}"
                f"__page_{page_number:02d}_of_{total_pages:02d}.png"
            )
            output_path = output_dir / output_name
            fig.savefig(output_path, dpi=160, bbox_inches="tight")
            print(f"[ĐÃ LƯU] {output_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hiển thị ảnh cần rà soát theo split và phân loài."
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Thư mục chứa các thư mục train, val và test.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=("test", "train", "val"),
        help="Các split cần hiển thị.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=4,
        help="Số cột trong mỗi contact sheet.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=16,
        help="Số ảnh tối đa trên mỗi trang.",
    )
    parser.add_argument(
        "--thumb-size",
        type=int,
        default=900,
        help="Kích thước tối đa của ảnh trước khi vẽ.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Lưu contact sheet vào thư mục này.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Không mở cửa sổ matplotlib; thường dùng cùng --save-dir.",
    )
    parser.add_argument(
        "--include-class-only",
        action="store_true",
        help=(
            "Hiển thị toàn bộ ảnh của những phân loài chỉ được ghi tên "
            "mà không có file/range cụ thể."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.dataset_root.is_dir():
        raise SystemExit(
            f"Không tìm thấy thư mục dataset: {args.dataset_root.resolve()}"
        )

    if args.cols < 1:
        raise SystemExit("--cols phải >= 1")

    if args.page_size < 1:
        raise SystemExit("--page-size phải >= 1")

    if args.thumb_size < 32:
        raise SystemExit("--thumb-size phải >= 32")

    if args.no_show and args.save_dir is None:
        raise SystemExit(
            "Dùng --no-show thì cần truyền thêm --save-dir để vẫn có đầu ra."
        )


def main() -> None:
    args = parse_args()
    validate_args(args)

    root = args.dataset_root.expanduser().resolve()
    save_dir = (
        args.save_dir.expanduser().resolve()
        if args.save_dir is not None
        else None
    )

    total_found = 0
    total_missing = 0

    for split in args.splits:
        split_dir = root / split

        if not split_dir.is_dir():
            print(f"[BỎ QUA] Không tìm thấy split: {split_dir}")
            continue

        split_targets = TARGETS.get(split, {})

        species_names = list(split_targets)
        if args.include_class_only:
            for species in CLASS_ONLY.get(split, []):
                if species not in species_names:
                    species_names.append(species)

        print(f"\n{'=' * 80}\nSPLIT: {split}\n{'=' * 80}")

        for species in species_names:
            species_dir = resolve_species_dir(split_dir, species)

            if species_dir is None:
                print(f"[THIẾU THƯ MỤC] {split}/{species}")
                total_missing += 1
                continue

            targets = split_targets.get(species)

            if targets:
                items, missing = collect_group(species_dir, targets)
            else:
                items = [(path, "") for path in image_files(species_dir)]
                missing = []

            print(
                f"[{split}/{species}] tìm thấy {len(items)} ảnh"
                + (f", thiếu {len(missing)} mục" if missing else "")
            )

            for missing_item in missing:
                print(f"  - [KHÔNG KHỚP] {missing_item}")

            total_found += len(items)
            total_missing += len(missing)

            if not items:
                continue

            show_group(
                split,
                species,
                items,
                cols=args.cols,
                page_size=args.page_size,
                thumb_size=args.thumb_size,
                save_dir=save_dir,
                show=not args.no_show,
            )

    print("\n" + "=" * 80)
    print(f"Tổng ảnh đã khớp: {total_found}")
    print(f"Tổng mục không khớp: {total_missing}")
    print("=" * 80)


if __name__ == "__main__":
    main()
