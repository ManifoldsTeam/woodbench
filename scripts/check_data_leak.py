#!/usr/bin/env python3
"""Check data leakage between train/val/test splits.

Usage examples:
  # If dataset already local
  python scripts/check_data_leak.py --data-root /path/to/data

  # If dataset is on Hugging Face Hub (set HUGGINGFACE_HUB_TOKEN env or login first)
  python scripts/check_data_leak.py --download-hf repo_id=owner/dataset --out /tmp/data

The script reports filename overlaps and identical-file hashes across splits,
and optionally checks a manifest CSV for augmented->original relationships.
"""
from __future__ import annotations

import argparse
import hashlib
import csv
import os
from pathlib import Path
from typing import Dict, List, Set

try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None


def md5_of_file(path: Path, chunk_size: int = 8192) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_images(data_root: Path, splits=("train", "val", "test")) -> Dict[str, List[Path]]:
    result = {s: [] for s in splits}
    for s in splits:
        root = data_root / s
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                result[s].append(p)
    return result


def find_overlaps(images_by_split: Dict[str, List[Path]]):
    # filename overlaps
    basename_map: Dict[str, Set[str]] = {}
    hash_map: Dict[str, Set[str]] = {}

    for split, paths in images_by_split.items():
        for p in paths:
            basename_map.setdefault(p.name, set()).add(split)

    # compute md5 per file (could be slow for large datasets)
    md5_cache: Dict[Path, str] = {}
    for split, paths in images_by_split.items():
        for p in paths:
            try:
                h = md5_of_file(p)
            except Exception:
                h = "<err>"
            md5_cache[p] = h
            hash_map.setdefault(h, set()).add(split)

    filename_overlaps = {fn: splits for fn, splits in basename_map.items() if len(splits) > 1}
    hash_overlaps = {h: splits for h, splits in hash_map.items() if len(splits) > 1}

    return filename_overlaps, hash_overlaps, md5_cache


def check_manifest(data_root: Path, manifest_names=("manifest.csv",)):
    # find manifest files under data_root
    manifests = []
    for name in manifest_names:
        for p in data_root.rglob(name):
            manifests.append(p)
    findings = []
    for m in manifests:
        try:
            with open(m, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    findings.append((m, row))
        except Exception:
            continue
    return manifests, findings


def summary_report(images_by_split, filename_overlaps, hash_overlaps, md5_cache, manifests):
    print("\n=== Summary ===")
    for s, imgs in images_by_split.items():
        print(f"{s}: {len(imgs)} images")
    print(f"Filename overlaps across splits: {len(filename_overlaps)}")
    print(f"Hash-identical files across splits: {len(hash_overlaps)}")

    if filename_overlaps:
        print("\nExamples of filename overlaps (basename -> splits):")
        for i, (fn, splits) in enumerate(filename_overlaps.items()):
            print(f"  {fn} -> {sorted(list(splits))}")
            if i >= 9:
                break

    if hash_overlaps:
        print("\nExamples of md5-identical overlaps (md5 -> splits):")
        for i, (h, splits) in enumerate(hash_overlaps.items()):
            print(f"  {h} -> {sorted(list(splits))}")
            # show file paths for the first example
            if i == 0:
                paths = [p for p, hv in md5_cache.items() if hv == h]
                for p in paths[:10]:
                    print(f"    {p}")
            if i >= 4:
                break

    if manifests:
        print(f"\nFound {len(manifests)} manifest file(s):")
        for m in manifests:
            print(f"  {m}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, help="Local dataset root (required if not downloading)")
    parser.add_argument("--download-hf", type=str, default=None,
                        help="If provided, use huggingface_hub snapshot_download with repo_id=owner/dataset")
    parser.add_argument("--out", type=str, default="./data",
                        help="Output dir for HF snapshot if --download-hf used")
    args = parser.parse_args()

    data_root = None
    if args.download_hf:
        if snapshot_download is None:
            raise SystemExit("huggingface_hub not available; install it to download from HF.")
        repo_id = args.download_hf
        outdir = Path(args.out).expanduser().resolve()
        print(f"Downloading snapshot of {repo_id} to {outdir} (this may take a while)...")
        
        data_root = outdir
    elif args.data_root:
        data_root = Path(args.data_root).expanduser().resolve()
    else:
        parser.error("Either --data-root or --download-hf must be provided")

    print(f"Using data root: {data_root}")
    images_by_split = collect_images(data_root)
    filename_overlaps, hash_overlaps, md5_cache = find_overlaps(images_by_split)
    manifests, manifest_entries = check_manifest(data_root)
    summary_report(images_by_split, filename_overlaps, hash_overlaps, md5_cache, manifests)

    if manifest_entries:
        print("\nManifest sample rows:")
        for i, (m, row) in enumerate(manifest_entries[:10]):
            print(f"  {m.name}: {row}")


if __name__ == "__main__":
    main()
