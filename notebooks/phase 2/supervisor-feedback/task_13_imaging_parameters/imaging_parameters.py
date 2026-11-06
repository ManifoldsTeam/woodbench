"""Collect the imaging facts available for the Methods section.

Supervisor request 5 (06-11-2026): imaging parameters for Methods.

Sources combined per species (17-species data/task-B-group-split):
  - the image files: pixel size, PNG mode/bit depth, PNG pHYs dpi tag;
  - docs/references/S2_Metadata.xlsx: collection date, origin, humidity,
    capture place, light conditions, capturer, device (merged cells expanded).
The camera / microscope / magnification are not in either source; see README.

Usage (from woodbench/):
  python "notebooks/phase 2/supervisor-feedback/task_13_imaging_parameters/imaging_parameters.py"
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import openpyxl
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "task_01_fivefold_cv_training"))
from origin_cv import PHASE1_MANIFEST, ROOT, load_origin_samples  # noqa: E402

METADATA = ROOT / "docs/references/S2_Metadata.xlsx"
# Spelling differences between the metadata sheet and the dataset folders.
NAME_FIX = {"Picea lauca": "Picea glauca", "Pinus  cernua": "Pinus cernua"}
FIELDS = {
    "Collected period": "collected",
    "Originin": "origin_place",
    "Humidity state": "humidity",
    "Captured place": "captured_place",
    "Light conditions": "light",
    "Capturers": "capturer",
    "Devices": "device",
}


def metadata_rows():
    sheet = openpyxl.load_workbook(METADATA, data_only=True).active
    grid = [[cell.value for cell in row] for row in sheet.iter_rows()]
    for merged in sheet.merged_cells.ranges:  # copy merged values into every covered cell
        value = grid[merged.min_row - 1][merged.min_col - 1]
        for r in range(merged.min_row - 1, merged.max_row):
            for c in range(merged.min_col - 1, merged.max_col):
                grid[r][c] = value
    header = grid[0]
    for values in grid[1:]:
        row = dict(zip(header, values))  # duplicate "TT" header: last one wins, unused
        if row.get("Science name"):
            yield row


def clean(value):
    if value is None:
        return ""
    if hasattr(value, "strftime"):
        return value.strftime("%b. %Y")
    return " ".join(str(value).replace("\xa0", " ").split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=ROOT / "data/task-B-group-split", type=Path)
    parser.add_argument(
        "--output_dir",
        default=ROOT / "artifacts/phase_2/supervisor-feedback/task_13_imaging_parameters",
        type=Path,
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    meta = defaultdict(lambda: defaultdict(set))
    meta_images = Counter()
    for row in metadata_rows():
        name = NAME_FIX.get(clean(row["Science name"]), clean(row["Science name"]))
        for column, key in FIELDS.items():
            if clean(row.get(column)):
                meta[name][key].add(clean(row[column]))
        number = clean(row["No of images"])
        meta_images[name] += int(float(number)) if number else 0

    files = defaultdict(Counter)
    counts = Counter()
    for record in load_origin_samples(args.data_root, PHASE1_MANIFEST, ("train", "val", "test")):
        image = Image.open(record["path"])
        width, height = image.size
        dpi = image.info.get("dpi")
        files[record["species"]]["size", f"{width}x{height}"] += 1
        files[record["species"]]["mode", f"{image.format} {image.mode}"] += 1
        files[record["species"]]["dpi", f"{round(dpi[0])}" if dpi else "none"] += 1
        counts[record["species"]] += 1

    def summary(counter, kind):
        items = sorted(((v, n) for (k, v), n in counter.items() if k == kind), key=lambda x: -x[1])
        return "; ".join(f"{v} ({n})" for v, n in items)

    rows = []
    for species in sorted(counts):
        width_values = {v for (k, v) in files[species] if k == "size"}
        rows.append({
            "species": species,
            "images_dataset": counts[species],
            "images_metadata": meta_images.get(species, ""),
            "pixel_format": "1280x1024" if all(v == "1280x1024" for v in width_values) else "2560 wide",
            "sizes": summary(files[species], "size"),
            "file": summary(files[species], "mode"),
            "png_dpi_tag": summary(files[species], "dpi"),
            **{key: " | ".join(sorted(meta[species][key])) for key in FIELDS.values()},
        })
    with open(args.output_dir / "imaging_by_species.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print({k: row[k] for k in ("species", "images_dataset", "images_metadata", "pixel_format", "collected",
                                   "captured_place", "light", "capturer", "device", "png_dpi_tag")})


if __name__ == "__main__":
    main()
