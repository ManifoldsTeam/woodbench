"""Group near-duplicate photos within Callitris, Cunninghamia and Fokienia.

Supervisor request 2 (06-11-2026): pHash near-duplicate grouping for these
three species, exported as an image -> group table for the data-collection team.

pHash alone misses the same wood surface photographed with a shifted frame, so
every within-species pair is also compared with ORB keypoints + RANSAC
(rotation/scale/translation). Two images are linked when either
  - pHash Hamming distance <= PHASH_MAX (10 bits is the closest pair of
    *different* species in task-B-group-split, so <= 9 is unambiguous), or
  - at least MIN_INLIERS geometrically consistent ORB matches, with a
    similarity transform of scale 0.9-1.1 (same magnification; any rotation).
Groups are connected components of these links, so a group can chain several
overlapping frames; `weakest_link_inliers` shows how strong its links are.

Threshold check (06-11-2026): 30,000 random cross-species pairs of these three
species never exceeded 12 inliers; visually checked pairs with 30-40 inliers
were real overlaps (shifted and/or rotated 90 degrees).

Usage (from woodbench/):
  python "notebooks/phase 2/supervisor-feedback/task_10_near_duplicate_grouping/near_duplicate_groups.py"
"""

import argparse
import csv
import itertools
import json
import sys
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "task_01_fivefold_cv_training"))
from origin_cv import PHASE1_MANIFEST, ROOT, load_origin_samples  # noqa: E402

SPECIES = ("Callitris columellaris", "Cunninghamia lanceolata", "Fokienia hodginsii")
PHASH_MAX = 9
MIN_INLIERS = 30
ORB_SIZE = (640, 512)
ORB_FEATURES = 1500
FONT = "/usr/share/fonts/liberation/LiberationSans-Regular.ttf"


def features(path):
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(gray, ORB_SIZE, interpolation=cv2.INTER_AREA)
    keypoints, descriptors = cv2.ORB_create(ORB_FEATURES).detectAndCompute(gray, None)
    points = np.float32([k.pt for k in keypoints]) if keypoints else np.zeros((0, 2), np.float32)
    phash = str(imagehash.phash(Image.open(path).convert("RGB")))
    return phash, points, descriptors


def inliers(a, b):
    """Return (RANSAC inliers, scale, rotation in degrees)."""
    (pa, da), (pb, db) = a, b
    if da is None or db is None or len(da) < 2 or len(db) < 2:
        return 0, 0.0, 0.0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return 0, 0.0, 0.0
    matrix, mask = cv2.estimateAffinePartial2D(
        pa[[m.queryIdx for m in good]],
        pb[[m.trainIdx for m in good]],
        method=cv2.RANSAC,
        ransacReprojThreshold=4,
    )
    if matrix is None:
        return 0, 0.0, 0.0
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    rotation = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
    return int(mask.sum()), scale, rotation


FEATS = {}


def compare(pair):
    i, j = pair
    return (i, j, *inliers(FEATS[i], FEATS[j]))


def components(n, edges):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in edges:
        parent[find(i)] = find(j)
    return [find(i) for i in range(n)]


def thumb(path, width=300):
    image = Image.open(path).convert("RGB")
    return image.resize((width, round(width * image.height / image.width)))


def group_sheet(rows, path, per_row=8, width=220):
    """One row per multi-image group: first `per_row` members, then a count."""
    font = ImageFont.truetype(FONT, 12)
    groups = {}
    for row in rows:
        if row["group_size"] > 1:
            groups.setdefault(row["group_id"], []).append(row)
    if not groups:
        return False
    height, pad = round(width * 0.8), 34
    sheet = Image.new(
        "RGB", (per_row * (width + 6) + 6, len(groups) * (height + pad + 6) + 6), "white"
    )
    draw = ImageDraw.Draw(sheet)
    for r, (gid, members) in enumerate(sorted(groups.items())):
        y = 6 + r * (height + pad + 6)
        draw.text(
            (6, y),
            f"{gid}: {len(members)} images, weakest link {members[0]['weakest_link_inliers']} inliers"
            + (f" (showing {per_row})" if len(members) > per_row else ""),
            fill="black",
            font=font,
        )
        for c, row in enumerate(members[:per_row]):
            x = 6 + c * (width + 6)
            draw.text((x, y + 15), f"{row['split']}/{row['file_name']}"[:34], fill="dimgray", font=font)
            sheet.paste(thumb(ROOT / row["path"], width).crop((0, 0, width, height)), (x, y + pad))
    sheet.save(path, quality=85)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=ROOT / "data/task-B-group-split", type=Path)
    parser.add_argument(
        "--output_dir",
        default=ROOT / "artifacts/phase_2/supervisor-feedback/task_10_near_duplicate_grouping",
        type=Path,
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = [
        r
        for r in load_origin_samples(args.data_root, PHASE1_MANIFEST, ("train", "val", "test"))
        if r["species"] in SPECIES
    ]
    records.sort(key=lambda r: (r["species"], Path(r["path"]).relative_to(args.data_root).as_posix()))
    with ProcessPoolExecutor(args.workers) as pool:
        feats = list(pool.map(features, [r["path"] for r in records], chunksize=8))
    FEATS.update({i: (f[1], f[2]) for i, f in enumerate(feats)})
    hashes = [imagehash.hex_to_hash(f[0]) for f in feats]

    pairs = [
        (i, j)
        for i, j in itertools.combinations(range(len(records)), 2)
        if records[i]["species"] == records[j]["species"]
    ]
    with ProcessPoolExecutor(args.workers) as pool:  # forked workers inherit FEATS
        scores = list(pool.map(compare, pairs, chunksize=256))

    best = {}
    links, pair_rows = [], []
    for i, j, n_inliers, scale, rotation in scores:
        distance = hashes[i] - hashes[j]
        if not 0.9 <= scale <= 1.1:
            n_inliers = 0
        for a, b in ((i, j), (j, i)):
            if a not in best or (n_inliers, -distance) > best[a][1:]:
                best[a] = (b, n_inliers, -distance)
        linked = distance <= PHASH_MAX or n_inliers >= MIN_INLIERS
        if linked:
            links.append((i, j, n_inliers))
        if linked or n_inliers >= 10:
            pair_rows.append(
                {
                    "species": records[i]["species"],
                    "image_a": Path(records[i]["path"]).relative_to(args.data_root).as_posix(),
                    "image_b": Path(records[j]["path"]).relative_to(args.data_root).as_posix(),
                    "orb_inliers": n_inliers,
                    "phash_distance": distance,
                    "scale": round(scale, 3),
                    "rotation_deg": round(rotation, 1),
                    "same_origin": records[i]["origin_key"] == records[j]["origin_key"],
                    "linked": linked,
                }
            )

    roots = components(len(records), [(i, j) for i, j, _ in links])
    sizes = {root: roots.count(root) for root in set(roots)}
    weakest = {}
    for i, j, n_inliers in links:
        weakest[roots[i]] = min(weakest.get(roots[i], 10**9), n_inliers)
    group_ids, counters = {}, {}
    rows = []
    for index, record in enumerate(records):
        species = record["species"]
        if roots[index] not in group_ids:
            counters[species] = counters.get(species, 0) + 1
            group_ids[roots[index]] = f"{species.split()[0][:4].upper()}-{counters[species]:03d}"
        partner, n_inliers, neg_distance = best[index]
        relative = Path(record["path"]).relative_to(args.data_root)
        rows.append(
            {
                "species": species,
                "split": relative.parts[0],
                "file_name": unicodedata.normalize("NFC", relative.name),
                "path": Path(record["path"]).resolve().relative_to(ROOT).as_posix(),
                "source_filename": unicodedata.normalize("NFC", record["source_filename"]),
                "origin_key": unicodedata.normalize("NFC", record["origin_key"]),
                "md5": record["md5"],
                "phash": str(hashes[index]),
                "group_id": group_ids[roots[index]],
                "group_size": sizes[roots[index]],
                "weakest_link_inliers": weakest.get(roots[index], ""),
                "closest_image": unicodedata.normalize(
                    "NFC", Path(records[partner]["path"]).relative_to(args.data_root).as_posix()
                ),
                "closest_orb_inliers": n_inliers,
                "closest_phash_distance": -neg_distance,
                "name_scheme": (
                    ("pơ mu" if "pơ mu" in unicodedata.normalize("NFC", relative.name) else "pơmu")
                    if species == "Fokienia hodginsii"
                    else ""
                ),
            }
        )

    # Fokienia has two file-name schemes that look like two photo sessions
    # (grey vs orange tone). Links across them are the least certain.
    mixed = {}
    for row in rows:
        if row["name_scheme"]:
            mixed.setdefault(row["group_id"], set()).add(row["name_scheme"])
    for row in rows:
        row["check_by_eye"] = len(mixed.get(row["group_id"], ())) > 1

    out = args.output_dir
    for name, data in (("image_groups.csv", rows), ("pair_scores.csv", pair_rows)):
        with (out / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    sheets = [
        name
        for species in SPECIES
        if group_sheet(
            [r for r in rows if r["species"] == species],
            out / (name := f"groups_{species.split()[0].lower()}.jpg"),
        )
    ]
    summary = {
        "data_root": args.data_root.resolve().relative_to(ROOT).as_posix(),
        "rule": f"link if pHash distance <= {PHASH_MAX} or ORB-RANSAC inliers >= {MIN_INLIERS} (scale 0.9-1.1); groups = connected components",
        "orb": {"resize": ORB_SIZE, "features": ORB_FEATURES, "ratio": 0.75, "ransac_px": 4},
        "pairs_compared": len(pairs),
        "species": {
            s: {
                "images": sum(r["species"] == s for r in rows),
                "groups": len({r["group_id"] for r in rows if r["species"] == s}),
                "multi_image_groups": len({r["group_id"] for r in rows if r["species"] == s and r["group_size"] > 1}),
                "images_in_multi_image_groups": sum(r["species"] == s and r["group_size"] > 1 for r in rows),
            }
            for s in SPECIES
        },
        "fokienia_groups_mixing_name_schemes": sum(len(v) > 1 for v in mixed.values()),
        "sheets": sheets,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
