"""Fig. 1: one representative image per species, 17 tiles, italic species names.

Supervisor request 4 (06-11-2026). Data: 17-species data/task-B-group-split.

Automatic choice per species (override with --override "Species=relative/path.png"):
  1. candidates: correct for all 5 CV v2 models (held-out predictions),
     no specular glare, sharpness >= species median;
  2. pick the candidate closest (cosine) to the species centroid of
     ImageNet-pretrained ConvNeXt-Tiny features (timm `convnext_tiny`, default
     in12k_ft_in1k weights, 224 cache).
     A generic backbone is used so the choice does not depend on our checkpoints.
A sheet with the 8 best candidates per species is written for manual review.

Usage (from woodbench/):
  python "notebooks/phase 2/supervisor-feedback/task_12_figure1_species_grid/figure1_species_grid.py"
"""

import argparse
import csv
import glob
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import timm  # noqa: E402
from timm.data import resolve_data_config  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont, ImageOps  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "task_01_fivefold_cv_training"))
from origin_cv import PHASE1_MANIFEST, ROOT, load_origin_samples  # noqa: E402

CV_DIR = ROOT / "artifacts/phase_2/supervisor-feedback/task_01_fivefold_cv_training/results-v2"
FONT = "/usr/share/fonts/liberation/LiberationSans-Regular.ttf"
TILE_ASPECT = 5 / 4  # width / height; all sources are 5:4 or slightly wider
COLUMNS = 6


def quality(path):
    gray = np.asarray(Image.open(path).convert("L"))
    gray = cv2.resize(gray, (1280, round(1280 * gray.shape[0] / gray.shape[1])), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()), float(100 * (gray >= 240).mean())


def cv_votes():
    files = sorted(glob.glob(str(CV_DIR / "*/fold_*/predictions.csv")))
    if len(files) != 25:
        raise RuntimeError(f"Expected 25 CV v2 prediction files, found {len(files)}")
    wrong, seen = defaultdict(int), defaultdict(int)
    for file in files:
        with open(file, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = Path(row["path"]).relative_to("data/task-B-group-split-224").as_posix()
                seen[key] += 1
                wrong[key] += row["predicted_species"] != row["true_species"]
    return wrong, seen


@torch.no_grad()
def embed(paths, cache_root):
    model = timm.create_model("convnext_tiny", pretrained=True, num_classes=0).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    config = resolve_data_config({}, model=model)
    mean = torch.tensor(config["mean"]).view(1, 3, 1, 1).to(device)
    std = torch.tensor(config["std"]).view(1, 3, 1, 1).to(device)
    features = []
    for start in range(0, len(paths), 128):
        batch = np.stack(
            [np.asarray(Image.open(cache_root / p).convert("RGB").resize((224, 224))) for p in paths[start:start + 128]]
        )
        x = torch.from_numpy(batch).permute(0, 3, 1, 2).float().div(255).to(device)
        features.append(torch.nn.functional.normalize(model((x - mean) / std), dim=1).cpu())
    return torch.cat(features).numpy()


def tile(path, width):
    return ImageOps.fit(Image.open(path).convert("RGB"), (width, round(width / TILE_ASPECT)), Image.LANCZOS)


def candidate_sheet(table, path, per_species=8, width=210):
    font = ImageFont.truetype(FONT, 12)
    bold = ImageFont.truetype(FONT.replace("Regular", "Bold"), 14)
    height = round(width / TILE_ASPECT)
    species = sorted({r["species"] for r in table})
    canvas = Image.new("RGB", (per_species * (width + 6) + 6, len(species) * (height + 42) + 6), "white")
    draw = ImageDraw.Draw(canvas)
    for r, name in enumerate(species):
        rows = sorted((x for x in table if x["species"] == name and x["candidate_rank"]), key=lambda x: x["candidate_rank"])
        y = 6 + r * (height + 42)
        draw.text((6, y), name, fill="black", font=bold)
        for c, row in enumerate(rows[:per_species]):
            x = 6 + c * (width + 6)
            draw.text((x, y + 18), f"#{row['candidate_rank']} {row['file_name']}"[:32],
                      fill="firebrick" if row["selected"] else "dimgray", font=font)
            canvas.paste(tile(ROOT / row["path"], width), (x, y + 34))
    canvas.save(path, quality=85)


def figure(selected, out, columns=COLUMNS):
    plt.rcParams.update({"font.family": "Liberation Sans", "pdf.fonttype": 42, "font.size": 7})
    species = sorted(selected)
    rows = -(-len(species) // columns)
    width_in = 7.0  # 178 mm, full journal page width
    gap = 0.06
    tile_w = (width_in - gap * (columns + 1)) / columns
    tile_h = tile_w / TILE_ASPECT
    label_h = 0.30
    height_in = rows * (tile_h + label_h) + 2 * gap
    fig = plt.figure(figsize=(width_in, height_in))
    for index, name in enumerate(species):
        row, col = divmod(index, columns)
        in_row = min(columns, len(species) - row * columns)
        offset = (columns - in_row) * (tile_w + gap) / 2  # centre a short last row
        left = gap + offset + col * (tile_w + gap)
        bottom = height_in - gap - (row + 1) * (tile_h + label_h)
        ax = fig.add_axes([left / width_in, (bottom + label_h) / height_in, tile_w / width_in, tile_h / height_in])
        ax.imshow(tile(ROOT / selected[name], 900))
        ax.set_axis_off()
        # Genus and epithet on fixed baselines so every label lines up.
        for line, text in enumerate(name.split(" ", 1)):
            fig.text((left + tile_w / 2) / width_in, (bottom + label_h - 0.11 - 0.11 * line) / height_in,
                     text, ha="center", va="baseline", style="italic")
    # PDF: vector text, photos resampled at 600 dpi (matplotlib default is 100).
    for suffix, kwargs in ((".pdf", {"dpi": 600}), (".png", {"dpi": 300})):
        fig.savefig(out / f"fig1_species_grid{suffix}", **kwargs)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=ROOT / "data/task-B-group-split", type=Path)
    parser.add_argument("--cache_root", default=ROOT / "data/task-B-group-split-224", type=Path)
    parser.add_argument(
        "--output_dir",
        default=ROOT / "artifacts/phase_2/supervisor-feedback/task_12_figure1_species_grid",
        type=Path,
    )
    parser.add_argument("--override", action="append", default=[],
                        help='"Species name=split/Species name/file.png" (relative to data_root)')
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    records = sorted(load_origin_samples(args.data_root, PHASE1_MANIFEST, ("train", "val", "test")),
                     key=lambda r: str(r["path"]))
    rel = [Path(r["path"]).relative_to(args.data_root).as_posix() for r in records]
    wrong, seen = cv_votes()
    if any(seen[p] != 5 for p in rel):
        raise RuntimeError("Every image needs 5 CV v2 predictions")
    with ProcessPoolExecutor(args.workers) as pool:
        stats = list(pool.map(quality, [r["path"] for r in records], chunksize=16))
    features = embed(rel, args.cache_root)

    by_species = defaultdict(list)
    for i, record in enumerate(records):
        by_species[record["species"]].append(i)
    table, selected = [], {}
    overrides = dict(item.split("=", 1) for item in args.override)
    for name, idx in sorted(by_species.items()):
        centroid = features[idx].mean(0)
        centroid /= np.linalg.norm(centroid)
        similarity = features[idx] @ centroid
        sharp_median = np.median([stats[i][0] for i in idx])
        eligible = [k for k, i in enumerate(idx)
                    if wrong[rel[i]] == 0 and stats[i][1] < 0.1 and stats[i][0] >= sharp_median]
        ranked = sorted(eligible, key=lambda k: -similarity[k])
        rank = {k: n + 1 for n, k in enumerate(ranked)}
        choice = overrides.get(name, rel[idx[ranked[0]]])
        if choice not in {rel[i] for i in idx}:
            raise ValueError(f"Override for {name} is not an image of that species: {choice}")
        selected[name] = Path(args.data_root.resolve().relative_to(ROOT)) / choice
        for k, i in enumerate(idx):
            table.append({
                "species": name,
                "file_name": Path(rel[i]).name,
                "path": (args.data_root.resolve().relative_to(ROOT) / rel[i]).as_posix(),
                "origin_key": records[i]["origin_key"],
                "cv_models_wrong": wrong[rel[i]],
                "sharpness": round(stats[i][0], 1),
                "glare_pct": round(stats[i][1], 3),
                "centroid_cosine": round(float(similarity[k]), 4),
                "candidate_rank": rank.get(k, ""),
                "selected": rel[i] == choice,
            })

    with open(out / "selection_table.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    candidate_sheet(table, out / "candidates.jpg")
    figure(selected, out)
    chosen = {
        name: {
            "path": path.as_posix(),
            "size": list(Image.open(ROOT / path).size),
            "override": name in overrides,
        }
        for name, path in selected.items()
    }
    (out / "selected.json").write_text(json.dumps(chosen, indent=2, ensure_ascii=False))
    print(json.dumps(chosen, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
