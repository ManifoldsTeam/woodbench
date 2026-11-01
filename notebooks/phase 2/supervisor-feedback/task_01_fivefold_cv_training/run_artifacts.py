"""Atomic artifacts and identity checks for fold-level restart."""

from contextlib import contextmanager
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid

import torch

ROOT = Path(__file__).resolve().parents[4]
PROTOCOL = "best-val-origin-cv-v2"
CHECKPOINT = "best.pth"
# Input degradations applied before augmentation, to train, val and held-out alike.
INPUT_VARIANTS = {"rgb": "", "grayscale": "grayscale3|", "px16": "resize224-16-224|"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def relative_path(path):
    return Path(os.path.relpath(Path(path).resolve(), ROOT)).as_posix()


def git_commit():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def make_config(args, dataset, splits):
    config = {
        name: getattr(args, name)
        for name in (
            "backbone",
            "n_folds",
            "epochs",
            "batch_size",
            "backbone_lr",
            "head_lr",
            "weight_decay",
            "label_smoothing",
            "seed",
            "use_90rotation",
            "num_workers",
            "warmup_epochs",
            "early_stopping_patience",
            "inner_val_fraction",
            "input_variant",
        )
    }
    config.update(
        protocol=PROTOCOL,
        image_size=224,
        pretrained=True,
        head_selector="classifier",
        amp=True,
        scheduler="linear-warmup+cosine",
        selection="best inner-val accuracy; held-out evaluated once with best.pth",
        augmentation=INPUT_VARIANTS[args.input_variant]
        + "resize224|hflip0.5|vflip0.5|"
        + ("random_90" if args.use_90rotation else "random_rotation30")
        + "|color_jitter0.2",
        manifest_path=relative_path(args.manifest_path),
        manifest_sha256=file_sha256(args.manifest_path),
        class_to_idx=dataset.class_to_idx,
    )
    plan = [
        {
            "path": Path(r["path"]).relative_to(dataset.data_root).as_posix(),
            "origin_key": r["origin_key"],
            "md5": r["md5"],
            "species": r["species"],
        }
        for r in dataset.records
    ]
    config["split_version"] = digest(
        {
            "records": plan,
            "held_out": [s["val_idx"].tolist() for s in splits],
            "inner_val": [s["inner_val_idx"].tolist() for s in splits],
        }
    )
    source_paths = [
        Path(__file__).parent / name
        for name in ("train_kfold.py", "origin_cv.py", "run_artifacts.py")
    ] + [ROOT / "src/data/transforms.py", ROOT / "src/train/optimizer.py"]
    config["code_sha256"] = digest(
        {relative_path(path): file_sha256(path) for path in source_paths}
    )
    return config


@contextmanager
def model_lock(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".training.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Training already owns {output_dir}") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def atomic_file(path):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_json(path, value):
    with atomic_file(path) as temporary:
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")


def save_checkpoint(path, value):
    with atomic_file(path) as temporary:
        torch.save(value, temporary)


def save_csv(path, rows):
    if not rows:
        raise ValueError("Cannot export empty rows")
    with atomic_file(path) as temporary:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def completed_fold(folder, fingerprint, verify_checksums=True):
    """Validate identity/artifacts; monitors can skip expensive checksum reads."""
    metrics_path = folder / "metrics.json"
    if not metrics_path.exists():
        return None
    metrics = json.loads(metrics_path.read_text())
    if (
        metrics.get("fingerprint") != fingerprint
        or metrics.get("status") != "completed"
    ):
        raise ValueError(
            f"Legacy or different config fingerprint: {metrics_path}; use another output_root"
        )
    if (
        metrics.get("fold") != int(folder.name.removeprefix("fold_"))
        or digest(metrics.get("config")) != fingerprint
        or metrics.get("checkpoint_path") != relative_path(folder / CHECKPOINT)
        or metrics.get("prediction_path") != relative_path(folder / "predictions.csv")
    ):
        raise ValueError(f"Completed fold identity mismatch: {folder}")
    for name in (CHECKPOINT, "predictions.csv"):
        artifact = folder / name
        if not artifact.is_file() or not artifact.stat().st_size:
            raise ValueError(f"Completed fold has missing artifact: {artifact}")
        if (
            verify_checksums
            and file_sha256(artifact) != metrics["artifact_sha256"][name]
        ):
            raise ValueError(f"Completed fold has changed artifact: {artifact}")
    return metrics


def prepare_run(output_dir, config, resume):
    fingerprint = digest(config)
    path = output_dir / "run_config.json"
    if path.exists() and json.loads(path.read_text()).get("fingerprint") != fingerprint:
        raise ValueError(
            "Different config/split fingerprint; choose another --output_root"
        )
    folders = list(output_dir.glob("fold_*"))
    if folders and not resume:
        raise FileExistsError(
            "Fold output exists; use --resume or another --output_root"
        )
    for folder in folders:
        completed_fold(folder, fingerprint)
        attempt = folder / "run.json"
        if (
            attempt.exists()
            and json.loads(attempt.read_text()).get("fingerprint") != fingerprint
        ):
            raise ValueError(f"Different unfinished fold fingerprint: {folder}")
    save_json(path, {"fingerprint": fingerprint, "config": config})
    return fingerprint


def process_token(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None
