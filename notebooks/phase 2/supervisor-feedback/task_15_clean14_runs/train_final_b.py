#!/usr/bin/env python3
"""Final-B on the 14 clean species, with the 06-11-2026 checkpoint rule.

Supervisor request 6 (06-11-2026): 5 models x 3 seeds on the fixed Final-B
split (data/task-B-group-split) without Callitris columellaris, Cunninghamia
lanceolata and Fokienia hodginsii: train 2,880 / val 803 / test 432 images,
140 origins.

Recipe = Final-B main (final_b_17class.EXPERIMENTS["main"]) with the changes
the supervisor asked for since 01-11-2026:

- backbone and head LR 1e-4 (01-11 LR selection chose 1e-4 for all models);
  head = model.get_classifier(), not the old substring rule;
- best.pth = lowest val loss over epochs warmup+1.. (warmup excluded), early
  stopping patience 15 on the same criterion. Same loop as train_kfold.py
  (its train_epoch/validate), so CV and Final-B follow one rule;
- val and test are scored once from best.pth after training.

Unchanged: AdamW (weight decay 1e-4), 5 epochs linear warmup + cosine, max 50
epochs, label smoothing 0.1, AMP, batch per backbone, src/data/transforms.py.

Data: the 224x224 cache (data/task-B-group-split-224) by default. Its files
are checked against docs/timeline/01-11-2026-task-B-group-split-224.sha256
before training. Full-resolution PNGs give the same tensors but make the CPU
the bottleneck (about 7 hours for the August Final-B runs).

A finished run (metrics.json present) is skipped. An unfinished run directory
is deleted and the run starts again from epoch 1; runs are never resumed.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "task_01_fivefold_cv_training"))
sys.path.insert(0, str(HERE.parent / "task_07_lr_selection"))

import timm  # noqa: E402
import torch  # noqa: E402

from origin_cv import SPECIES_SETS  # noqa: E402
from run_artifacts import file_sha256, save_checkpoint, save_csv, save_json, utc_now  # noqa: E402
from train_kfold import SELECTION_METRICS, train_epoch, validate  # noqa: E402
from train_lr_select import BACKBONE_CONFIGS, MAIN, score  # noqa: E402
from src.data import get_dataloaders  # noqa: E402
from src.models import create_model  # noqa: E402
from src.models.factory import BACKBONE_NAMES  # noqa: E402
from src.train import (  # noqa: E402
    WandbTracker,
    create_loss_function,
    create_optimizer,
    create_scheduler,
    split_head_parameters,
)
from src.utils import set_seed  # noqa: E402

HEAD_SELECTOR = "classifier"
SELECTION_METRIC = "val_loss"
CHECKSUMS = ROOT / "docs/timeline/01-11-2026-task-B-group-split-224.sha256"
SPLIT_SIZES = {"clean14": {"train": 2880, "val": 803, "test": 432}}
CODE_FILES = [
    Path(__file__),
    HERE.parent / "task_01_fivefold_cv_training/train_kfold.py",
    HERE.parent / "task_01_fivefold_cv_training/origin_cv.py",
    HERE.parent / "task_07_lr_selection/train_lr_select.py",
    ROOT / "src/data/transforms.py",
    ROOT / "src/data/dataset.py",
    ROOT / "src/train/optimizer.py",
]


def relative(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def git_state():
    def git(*args):
        try:
            return subprocess.run(
                ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain", "--untracked-files=no")
    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(status) if status is not None else None}


def verify_data(data_root, checksums):
    """Every image listed in checksums matches, and data_root holds nothing else.

    The list stores paths as task-B-group-split-224/<split>/<species>/<file>;
    the first component is replaced by data_root.
    """
    data_root = Path(data_root)
    expected = {}
    for line in Path(checksums).read_text(encoding="utf-8").splitlines():
        digest, path = line.split("  ", 1)
        expected[Path(*Path(path).parts[1:]).as_posix()] = digest
    present = {
        p.relative_to(data_root).as_posix()
        for p in data_root.rglob("*")
        if p.is_file() and p.suffix.lower() == ".png"
    }
    if present != set(expected):
        missing, extra = sorted(set(expected) - present), sorted(present - set(expected))
        raise ValueError(f"Data mismatch in {data_root}: missing {missing[:3]}, extra {extra[:3]}")
    bad = [name for name, digest in expected.items() if file_sha256(data_root / name) != digest]
    if bad:
        raise ValueError(f"{len(bad)} images differ from {checksums}: {bad[:3]}")
    return hashlib.sha256(Path(checksums).read_bytes()).hexdigest()


def run_dir(output_root, seed, backbone):
    return Path(output_root) / f"seed_{seed}" / backbone


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backbone", required=True, choices=list(BACKBONE_CONFIGS))
    parser.add_argument("--seed", type=int, required=True, help="Final-B main: 42, 123, 456")
    parser.add_argument("--species_set", choices=list(SPECIES_SETS), default="clean14")
    parser.add_argument("--backbone_lr", type=float, default=1e-4)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--early_stopping_patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, help="Default: Final-B batch per backbone")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_root", type=Path, default=ROOT / "data/task-B-group-split-224")
    parser.add_argument(
        "--checksums",
        type=Path,
        default=CHECKSUMS,
        help="sha256 list of the data; 'none' skips the check (smoke tests only)",
    )
    parser.add_argument(
        "--output_root", type=Path, default=ROOT / "data/experiments/Task-B-clean14-final"
    )
    parser.add_argument("--wandb_project", default="woodbench-clean14")
    parser.add_argument("--dryrun", action="store_true", help="Check data, model and groups only")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.batch_size = args.batch_size or BACKBONE_CONFIGS[args.backbone]["batch_size"]
    if not 0 <= args.warmup_epochs < args.epochs or args.early_stopping_patience < 0:
        raise ValueError("warmup_epochs must be in [0, epochs); patience nonnegative")
    output_dir = run_dir(args.output_root, args.seed, args.backbone)
    if (output_dir / "metrics.json").exists() and not args.dryrun:
        print(f"Already completed, skipping: {output_dir}")
        return json.loads((output_dir / "metrics.json").read_text())

    data_sha256 = None
    if str(args.checksums) != "none":
        data_sha256 = verify_data(args.data_root, args.checksums)
        print(f"Data OK: {args.data_root} matches {relative(args.checksums)}")
    excluded = SPECIES_SETS[args.species_set]
    classes = sorted(
        p.name for p in (args.data_root / "train").iterdir() if p.is_dir() and p.name not in excluded
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        task="taskB",
        use_90rotation=bool(MAIN["use_90rotation"]),
        classes=classes,
    )
    sizes = {
        "train": len(train_loader.dataset),
        "val": len(val_loader.dataset),
        "test": len(test_loader.dataset),
    }
    if data_sha256 and SPLIT_SIZES.get(args.species_set, sizes) != sizes:
        raise ValueError(f"Unexpected split sizes {sizes}")
    model = create_model(args.backbone, num_classes=len(classes), pretrained=not args.dryrun)
    head, backbone = split_head_parameters(model, HEAD_SELECTOR)
    run_id = f"{args.backbone}-s{args.seed}-{args.species_set}-{uuid.uuid4().hex}"
    config = {
        "task": "Task-B",
        "step": "final-b-clean14",
        "run_id": run_id,
        "backbone": args.backbone,
        "timm_name": BACKBONE_NAMES[args.backbone],
        "seed": args.seed,
        "species_set": args.species_set,
        "excluded_species": sorted(excluded),
        "classes": classes,
        "split_sizes": sizes,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "backbone_lr": args.backbone_lr,
        "head_lr": args.head_lr,
        "head_selector": HEAD_SELECTOR,
        "head_params": sum(p.numel() for p in head),
        "backbone_params": sum(p.numel() for p in backbone),
        "optimizer": "AdamW",
        "weight_decay": 1e-4,
        "label_smoothing": float(MAIN["label_smoothing"]),
        "use_90rotation": bool(MAIN["use_90rotation"]),
        "scheduler": "linear-warmup+cosine",
        "warmup_epochs": args.warmup_epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "selection_metric": SELECTION_METRIC,
        "select_after_warmup": True,
        "selection": f"lowest val loss, epochs {args.warmup_epochs + 1}+ only (warmup excluded); "
        "val and test scored once from best.pth",
        "amp": device.type == "cuda",
        "pretrained": "ImageNet (timm default weights)",
        "num_workers": args.num_workers,
        "data_root": relative(args.data_root),
        "data_checksums": relative(args.checksums) if data_sha256 else None,
        "data_checksums_sha256": data_sha256,
        "code_sha256": {relative(p): file_sha256(p) for p in CODE_FILES},
        **git_state(),
        "versions": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),  # TorchVersion breaks weights_only loading
            "timm": timm.__version__,
        },
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "hostname": platform.node(),
    }
    print(
        f"[{args.backbone} seed {args.seed}] {len(classes)} species {sizes}; head LR "
        f"{args.head_lr} on {config['head_params']:,} params, backbone LR {args.backbone_lr} "
        f"on {config['backbone_params']:,}; batch {args.batch_size}"
    )
    if args.dryrun:
        images, _ = next(iter(val_loader))
        with torch.no_grad():
            logits = model.eval()(images[:2])
        assert logits.shape == (2, len(classes)), logits.shape
        print("DRYRUN PASSED: data, classes, model, parameter groups and forward pass")
        return config

    if output_dir.exists():
        print(f"Removing unfinished run (no metrics.json): {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    config["started_at"] = utc_now()
    save_json(output_dir / "run_config.json", config)

    model.to(device)
    optimizer = create_optimizer(
        model,
        lr=args.head_lr,
        weight_decay=config["weight_decay"],
        backbone_lr=args.backbone_lr,
        head_selector=HEAD_SELECTOR,
    )
    scheduler = create_scheduler(
        optimizer, epochs=args.epochs, scheduler_type="cosine", warmup_epochs=args.warmup_epochs
    )
    criterion = create_loss_function("label_smoothing", label_smoothing=config["label_smoothing"])
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type) if use_amp else None
    tracker = None
    if os.getenv("WANDB_API_KEY"):
        tracker = WandbTracker.start(
            project=args.wandb_project,
            name=f"final-{args.backbone}-s{args.seed}-{args.species_set}",
            config=config,
            dir_path=str(output_dir),
        )
    better = SELECTION_METRICS[SELECTION_METRIC]
    first_eligible = args.warmup_epochs + 1
    history, best, stale = [], None, 0
    try:
        for epoch in range(args.epochs):
            loss, accuracy = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
            val_loss, val_accuracy = validate(model, val_loader, criterion, device, use_amp)
            lr = {group.get("name", "all"): group["lr"] for group in optimizer.param_groups}
            scheduler.step()
            eligible = epoch + 1 >= first_eligible
            improved = eligible and (best is None or better(val_loss, best["val_loss"]))
            if improved:
                best = {"epoch": epoch + 1, "val_loss": val_loss, "val_accuracy": val_accuracy}
                stale = 0
                save_checkpoint(
                    output_dir / "best.pth",
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch + 1,
                        "run_id": run_id,
                        "config": config,
                        "val_loss": val_loss,
                        "val_accuracy": val_accuracy,
                    },
                )
            elif eligible:
                stale += 1
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": loss,
                    "train_accuracy": accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "lr": lr,
                    "best": improved,
                    "eligible": eligible,
                }
            )
            save_json(output_dir / "history.json", history)
            if tracker:
                tracker.log(history[-1], step=epoch + 1)
            print(
                f"[{args.backbone} s{args.seed}] epoch {epoch + 1}/{args.epochs}: "
                f"train={accuracy:.4f} val={val_accuracy:.4f} loss={val_loss:.4f}"
                f"{' [best]' if improved else ''}{'' if eligible else ' (warmup)'}",
                flush=True,
            )
            if args.early_stopping_patience and stale >= args.early_stopping_patience:
                print(f"Early stopping: {stale} epochs without lower val loss")
                break
    finally:
        if tracker:
            tracker.finish()

    checkpoint = torch.load(output_dir / "best.pth", map_location="cpu", weights_only=True)
    if checkpoint["run_id"] != run_id or checkpoint["epoch"] != best["epoch"]:
        raise ValueError("best.pth does not belong to this run")
    model.load_state_dict(checkpoint["model_state_dict"])
    val, val_rows = score(model, val_loader, criterion, device)
    test, test_rows = score(model, test_loader, criterion, device)
    save_csv(output_dir / "predictions_val.csv", val_rows)
    save_csv(output_dir / "predictions_test.csv", test_rows)
    metrics = {
        **{key: config[key] for key in ("run_id", "backbone", "seed", "species_set", "backbone_lr", "head_lr")},
        "selection": config["selection"],
        "best_epoch": best["epoch"],
        "best_val_loss_training": best["val_loss"],
        "epochs_trained": len(history),
        "stopped_early": len(history) < args.epochs,
        "val": val,
        "test": test,
        "finished_at": utc_now(),
        "artifact_sha256": {
            name: file_sha256(output_dir / name)
            for name in ("best.pth", "predictions_val.csv", "predictions_test.csv")
        },
    }
    save_json(output_dir / "metrics.json", metrics)  # completion marker, written last
    print(
        f"[{args.backbone} s{args.seed}] best epoch {best['epoch']}/{len(history)}: "
        f"val acc {val['accuracy'] * 100:.2f}% loss {val['loss']:.4f} | "
        f"test acc {test['accuracy'] * 100:.2f}% F1 {test['f1_macro'] * 100:.2f}%"
    )
    return metrics


if __name__ == "__main__":
    main()
