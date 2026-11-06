"""Inspect the two most-confused specimens: 3384.4 (P. krempfii), 3393.1 (P. dalatensis).

Supervisor request 3 (06-11-2026): check whether these specimens are
mislabelled, blurred or otherwise abnormal.

Evidence collected (17-species task-B-group-split, CV v2 held-out predictions):
  1. per-image error counts over the 5 CV v2 models (each origin is held out once);
  2. image statistics per origin: size, sharpness (variance of Laplacian),
     brightness, contrast, mean hue/saturation;
  3. ORB + RANSAC overlap search of each specimen against every image of the
     confused species and of its own species (same matcher as task 10), to see
     whether any photo is a frame of a specimen with the other label;
  4. contact sheets for visual inspection.

Usage (from woodbench/):
  python "notebooks/phase 2/supervisor-feedback/task_11_specimen_inspection/inspect_specimens.py"
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
from PIL import Image, ImageDraw, ImageFont, ImageOps  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "task_01_fivefold_cv_training"))
sys.path.insert(0, str(HERE.parent / "task_10_near_duplicate_grouping"))
from origin_cv import PHASE1_MANIFEST, ROOT, load_origin_samples  # noqa: E402
import near_duplicate_groups as nd  # noqa: E402

TARGETS = {
    "Pinus krempfii::3384. Pinus krempfii.4": "Pinus dalatensis",
    "Pinus dalatensis::3393. Pinus dalatensis.1": "Pinus krempfii",
}
SPECIES = ("Pinus krempfii", "Pinus dalatensis")
CV_DIR = ROOT / "artifacts/phase_2/supervisor-feedback/task_01_fivefold_cv_training/results-v2"
STAT_WIDTH = 1280
FONT = nd.FONT


def short(origin_key):
    """'Pinus krempfii::3384. Pinus krempfii.4' -> '3384.4'"""
    stem = origin_key.split("::")[1]
    number, rest = stem.split(". ", 1)
    suffix = rest.rsplit(".", 1)[1] if rest.count(".") else ""
    return f"{number}.{suffix}" if suffix else number


def is_target(species, origin):
    return any(t.startswith(f"{species}::") and short(t) == origin for t in TARGETS)


def origin_order(key):
    species, origin = key
    suffix = origin.split(".")[1] if "." in origin else "0"
    return species, int(suffix)


def image_stats(path):
    rgb = Image.open(path).convert("RGB")
    width, height = rgb.size
    small = np.asarray(rgb.resize((STAT_WIDTH, round(STAT_WIDTH * height / width)), Image.BILINEAR))
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    return {
        "width": width,
        "height": height,
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "hue": float(hsv[..., 0].mean()),
        "saturation": float(hsv[..., 1].mean()),
        # Specular glare: near-white pixels, in percent of the frame.
        "glare_pct": float(100 * (gray >= 240).mean()),
    }


def cv_predictions():
    """relative path -> list of (model, predicted_species, confidence)."""
    out = defaultdict(list)
    files = sorted(glob.glob(str(CV_DIR / "*/fold_*/predictions.csv")))
    if len(files) != 25:
        raise RuntimeError(f"Expected 25 CV v2 prediction files, found {len(files)}")
    for file in files:
        model = Path(file).parts[-3]
        with open(file, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["true_species"] in SPECIES:
                    relative = Path(row["path"]).relative_to("data/task-B-group-split-224").as_posix()
                    out[relative].append((model, row["predicted_species"], float(row["confidence"])))
    return out


def compare(pair):
    i, j = pair
    return (i, j, *nd.inliers(nd.FEATS[i], nd.FEATS[j]))


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sheet(groups, path, width=240, per_row=8):
    """groups: list of (title, [(image_path, caption)])."""
    font = ImageFont.truetype(FONT, 12)
    bold = ImageFont.truetype(FONT.replace("Regular", "Bold"), 15)
    height = round(width * 0.8)
    rows_needed = sum(-(-len(items) // per_row) for _, items in groups)
    canvas = Image.new(
        "RGB",
        (per_row * (width + 6) + 6, rows_needed * (height + 22) + len(groups) * 26 + 6),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    y = 6
    for title, items in groups:
        draw.text((6, y), title, fill="black", font=bold)
        y += 26
        for k, (image_path, caption) in enumerate(items):
            x = 6 + (k % per_row) * (width + 6)
            image = ImageOps.fit(Image.open(image_path).convert("RGB"), (width, height))
            canvas.paste(image, (x, y + 16))
            draw.text((x, y), caption[:38], fill="firebrick" if "wrong" in caption else "dimgray", font=font)
            if k % per_row == per_row - 1 or k == len(items) - 1:
                y += height + 22
    canvas.save(path, quality=85)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=ROOT / "data/task-B-group-split", type=Path)
    parser.add_argument(
        "--output_dir",
        default=ROOT / "artifacts/phase_2/supervisor-feedback/task_11_specimen_inspection",
        type=Path,
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    records = [
        r
        for r in load_origin_samples(args.data_root, PHASE1_MANIFEST, ("train", "val", "test"))
        if r["species"] in SPECIES
    ]
    records.sort(key=lambda r: str(r["path"]))
    rel = [Path(r["path"]).relative_to(args.data_root).as_posix() for r in records]
    predictions = cv_predictions()
    missing = [p for p in rel if len(predictions.get(p, ())) != 5]
    if missing:
        raise RuntimeError(f"{len(missing)} images lack 5 CV v2 predictions, e.g. {missing[0]}")

    with ProcessPoolExecutor(args.workers) as pool:
        stats = list(pool.map(image_stats, [r["path"] for r in records], chunksize=8))

    # 1 + 2: per-image table for both species (other origins are the reference).
    images = []
    for record, relative, stat in zip(records, rel, stats):
        votes = predictions[relative]
        wrong = [(m, s, c) for m, s, c in votes if s != record["species"]]
        images.append(
            {
                "species": record["species"],
                "origin": short(record["origin_key"]),
                "split": relative.split("/")[0],
                "file_name": Path(relative).name,
                "models_wrong": len(wrong),
                "wrong_predictions": "; ".join(f"{m}:{s}({c:.2f})" for m, s, c in wrong),
                "mean_confidence": round(float(np.mean([c for _, _, c in votes])), 3),
                **{k: round(v, 2) if isinstance(v, float) else v for k, v in stat.items()},
                "path": Path(record["path"]).resolve().relative_to(ROOT).as_posix(),
                "md5": record["md5"],
            }
        )
    write_csv(out / "image_table.csv", images)

    by_origin = defaultdict(list)
    for row in images:
        by_origin[(row["species"], row["origin"])].append(row)
    origin_rows = []
    for (species, origin), rows in sorted(by_origin.items()):
        heights = sorted({r["height"] for r in rows})
        origin_rows.append(
            {
                "species": species,
                "origin": origin,
                "target": is_target(species, origin),
                "split": ",".join(sorted({r["split"] for r in rows})),
                "images": len(rows),
                "cv_error_rate_pct": round(100 * sum(r["models_wrong"] for r in rows) / (5 * len(rows)), 1),
                "images_wrong_by_3plus_models": sum(r["models_wrong"] >= 3 for r in rows),
                "height_min": heights[0],
                "height_max": heights[-1],
                **{
                    f"{k}_median": round(float(np.median([r[k] for r in rows])), 2)
                    for k in ("sharpness", "brightness", "contrast", "hue", "saturation", "glare_pct")
                },
            }
        )
    write_csv(out / "origin_table.csv", origin_rows)

    # 3: overlap search of each target specimen against both species.
    with ProcessPoolExecutor(args.workers) as pool:
        feats = list(pool.map(nd.features, [r["path"] for r in records], chunksize=8))
    nd.FEATS.update({i: (f[1], f[2]) for i, f in enumerate(feats)})
    pairs = []
    for i, record in enumerate(records):
        if record["origin_key"] in TARGETS:
            pairs += [
                (i, j)
                for j, other in enumerate(records)
                if other["origin_key"] != record["origin_key"]
                and (j > i or records[j]["origin_key"] not in TARGETS)
            ]
    with ProcessPoolExecutor(args.workers) as pool:
        scores = list(pool.map(compare, pairs, chunksize=128))
    overlap_rows = []
    best = defaultdict(lambda: defaultdict(int))
    for i, j, n_inliers, scale, rotation in scores:
        if not 0.9 <= scale <= 1.1:
            n_inliers = 0
        if records[j]["origin_key"] in TARGETS and records[i]["origin_key"] not in TARGETS:
            i, j = j, i
        key = (short(records[i]["origin_key"]), records[j]["species"], short(records[j]["origin_key"]))
        best[key[0]][key[1:]] = max(best[key[0]][key[1:]], n_inliers)
        if n_inliers >= 15:
            overlap_rows.append(
                {
                    "target_image": rel[i],
                    "other_image": rel[j],
                    "other_species": records[j]["species"],
                    "other_origin": short(records[j]["origin_key"]),
                    "orb_inliers": n_inliers,
                    "scale": round(scale, 3),
                    "rotation_deg": round(rotation, 1),
                }
            )
    overlap_rows.sort(key=lambda r: -r["orb_inliers"])
    if overlap_rows:
        write_csv(out / "overlap_pairs.csv", overlap_rows)
    overlap_summary = {
        target: {f"{s} {o}": n for (s, o), n in sorted(found.items(), key=lambda kv: -kv[1])}
        for target, found in best.items()
    }

    # 4a: metric plot per origin.
    metrics = ("sharpness", "brightness", "contrast", "saturation", "glare_pct", "height")
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 2.6 * len(metrics)))
    order = sorted(by_origin, key=origin_order)
    labels = [f"{'PK' if s.endswith('krempfii') else 'PD'} {o}" for s, o in order]
    for ax, metric in zip(axes, metrics):
        data = [[r[metric] for r in by_origin[k]] for k in order]
        box = ax.boxplot(data, patch_artist=True, showfliers=True)
        for patch, key in zip(box["boxes"], order):
            patch.set_facecolor("tomato" if is_target(*key) else ("lightsteelblue" if key[0].endswith("krempfii") else "wheat"))
        ax.set_ylabel(metric)
        ax.set_xticks(range(1, len(order) + 1), labels if metric == metrics[-1] else [""] * len(order), rotation=60)
    axes[0].set_title("Per-origin image statistics (red = inspected specimen; blue = P. krempfii, beige = P. dalatensis)")
    fig.tight_layout()
    fig.savefig(out / "origin_statistics.png", dpi=110)
    plt.close(fig)

    # 4b: contact sheets.
    rng = np.random.default_rng(0)
    for target, confused in TARGETS.items():
        own_species = target.split("::")[0]
        name = short(target)
        rows = sorted(by_origin[(own_species, name)], key=lambda r: (-r["models_wrong"], r["file_name"]))
        groups = [
            (
                f"{own_species} {name} (label) - all {len(rows)} images, sorted by CV v2 errors",
                [
                    (ROOT / r["path"], f"{r['models_wrong']}/5 wrong  {r['file_name']}" if r["models_wrong"]
                     else f"0/5  {r['file_name']}")
                    for r in rows
                ],
            )
        ]
        for species, title in ((own_species, f"{own_species}, other origins (reference)"),
                               (confused, f"{confused} (predicted species), all origins")):
            pool_rows = [r for r in images if r["species"] == species and r["origin"] != name]
            pick = rng.choice(len(pool_rows), 16, replace=False)
            groups.append((title + " - 16 random", [
                (ROOT / pool_rows[k]["path"], f"{pool_rows[k]['origin']}  {pool_rows[k]['file_name']}")
                for k in sorted(pick, key=lambda k: pool_rows[k]["origin"])
            ]))
        sheet(groups, out / f"sheet_{name}.jpg")

    targets = [r for r in origin_rows if r["target"]]
    summary = {
        "data_root": args.data_root.resolve().relative_to(ROOT).as_posix(),
        "predictions": CV_DIR.relative_to(ROOT).as_posix() + "/*/fold_*/predictions.csv (25 files)",
        "targets": targets,
        "max_orb_inliers_by_origin": overlap_summary,
        "overlap_pairs_ge_15_inliers": len(overlap_rows),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
