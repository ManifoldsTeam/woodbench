#!/usr/bin/env python3
"""Pre-resize a split folder to the pipeline input size, losslessly.

Every train/eval transform in src/data/transforms.py starts with
Resize((224, 224)) on an RGB PIL image. Applying that exact op once and saving
PNG gives the same pixels the model would see, so later epochs only decode
224x224 images. Relative paths and file names are kept, so the origin manifest
(origin_cv.py) and class folders still match.

Usage:
    python scripts/build_resized_cache.py data/task-B-group-split data/task-B-group-split-224
    python scripts/build_resized_cache.py SRC DST --verify 200   # tensor check only
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import os
from pathlib import Path
import random

import torch
from PIL import Image
from torchvision import transforms as T

SIZE = 224
RESIZE = T.Resize((SIZE, SIZE))
EXTENSIONS = {".png", ".jpg", ".jpeg"}


def resize_one(job):
    src, dst = job
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        RESIZE(image.convert("RGB")).save(dst, format="PNG")
    return dst


def verify(src_root, dst_root, files, n):
    """Pixels the model sees: original -> Resize vs cached -> Resize (identity)."""
    to_tensor = T.Compose([T.Resize((SIZE, SIZE)), T.PILToTensor()])
    sample = random.Random(0).sample(files, min(n, len(files)))
    for path in sample:
        rel = path.relative_to(src_root)
        with Image.open(path) as a, Image.open(dst_root / rel) as b:
            if not torch.equal(to_tensor(a.convert("RGB")), to_tensor(b.convert("RGB"))):
                raise SystemExit(f"MISMATCH: {rel}")
    print(f"VERIFY OK: {len(sample)} random images give identical uint8 tensors")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    parser.add_argument("--verify", type=int, default=200, help="Images to compare")
    args = parser.parse_args()

    files = sorted(
        p for p in args.src.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS
    )
    jobs = [(p, args.dst / p.relative_to(args.src)) for p in files]
    todo = [job for job in jobs if not job[1].exists()]
    print(f"{len(files)} images; {len(todo)} to resize with {args.workers} workers")
    with ProcessPoolExecutor(args.workers) as pool:
        for i, _ in enumerate(pool.map(resize_one, todo, chunksize=16), 1):
            if i % 500 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    verify(args.src, args.dst, files, args.verify)

    # Checksums of the cache, so later runs can verify it like the source split.
    lines = []
    for p in sorted(args.dst.rglob("*")):
        if p.is_file() and p.suffix.lower() in EXTENSIONS:
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            lines.append(f"{digest}  {p.relative_to(args.dst.parent).as_posix()}")
    (args.dst.parent / f"{args.dst.name}.sha256").write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} checksums to {args.dst.parent / (args.dst.name + '.sha256')}")


if __name__ == "__main__":
    main()
