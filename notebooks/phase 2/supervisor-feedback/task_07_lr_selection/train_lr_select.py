#!/usr/bin/env python3
"""Step 1: compare backbone learning rates on the Final-B validation split.

One run trains one backbone with one backbone LR on data/task-B-group-split
(train 3,365 / val 903 / test 532), with the Phase 1 Final-B main recipe
(final_b_17class.train_one): same transforms, AdamW, linear warmup + cosine,
AMP, label smoothing 0.1, early stopping on val accuracy (patience 10), batch
size per backbone. Two differences, both intended:

- Head parameters are the true classifier (model.get_classifier()), not the
  historical substring rule, so "backbone LR" means the whole backbone.
- val and test are scored from the best checkpoint after training. Selection
  uses val only; test is recorded but must not drive the choice.

Runs are never resumed: an interrupted run must be deleted and started again.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks/phase 1/task-B-17-new"))

import timm
import torch
from sklearn.metrics import f1_score

from final_b_17class import (
    BACKBONE_CONFIGS,
    EXPERIMENTS,
    prepare_environment,
    verify_17class_dataset,
)
from src.data import get_dataloaders
from src.models import create_model
from src.models.factory import BACKBONE_NAMES
from src.train import (
    TrainingEngine,
    WandbTracker,
    create_loss_function,
    create_optimizer,
    create_scheduler,
    split_head_parameters,
)
from src.utils import set_seed

HEAD_SELECTOR = "classifier"
MAIN = EXPERIMENTS["main"]


def lr_tag(lr):
    """1e-05 -> '1e-5', used in output folder names."""
    mantissa, exponent = f"{lr:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def run_dir(output_root, backbone_lr, seed, backbone):
    return Path(output_root) / f"bblr_{lr_tag(backbone_lr)}" / f"seed_{seed}" / backbone


def git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def relative(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


@torch.no_grad()
def score(model, loader, criterion, device):
    """FP32 loss (mean per image), accuracy, macro/per-class F1 and predictions."""
    model.eval()
    classes = list(loader.dataset.classes)
    labels, predictions, confidences, loss_sum = [], [], [], 0.0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss_sum += criterion(outputs, targets).item() * len(targets)
        probabilities = outputs.softmax(1)
        confidence, predicted = probabilities.max(1)
        labels.extend(targets.tolist())
        predictions.extend(predicted.tolist())
        confidences.extend(confidence.tolist())
    indices = list(range(len(classes)))
    per_class = f1_score(labels, predictions, labels=indices, average=None, zero_division=0)
    metrics = {
        "loss": loss_sum / len(labels),
        "accuracy": sum(a == b for a, b in zip(labels, predictions)) / len(labels),
        "f1_macro": float(
            f1_score(labels, predictions, labels=indices, average="macro", zero_division=0)
        ),
        "f1_per_class": {name: float(value) for name, value in zip(classes, per_class)},
        "n": len(labels),
    }
    rows = [
        {
            "path": relative(path),
            "true_label": label,
            "predicted_label": prediction,
            "true_species": classes[label],
            "predicted_species": classes[prediction],
            "confidence": confidence,
        }
        for (path, _), label, prediction, confidence in zip(
            loader.dataset.samples, labels, predictions, confidences
        )
    ]
    return metrics, rows


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backbone", required=True, choices=list(BACKBONE_CONFIGS))
    parser.add_argument("--backbone_lr", type=float, required=True)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, help="Default: Final-B batch per backbone")
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_root", type=Path, default=ROOT / "data/task-B-group-split")
    parser.add_argument(
        "--output_root", type=Path, default=ROOT / "data/experiments/Task-B-lr-select"
    )
    parser.add_argument("--wandb_project", default="woodbench-lr-selection")
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Check data, model and parameter groups without training",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.batch_size = args.batch_size or BACKBONE_CONFIGS[args.backbone]["batch_size"]
    if not 0 <= args.warmup_epochs < args.epochs:
        raise ValueError("warmup_epochs must be in [0, epochs)")
    prepare_environment(ROOT)
    verify_17class_dataset(args.data_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = run_dir(args.output_root, args.backbone_lr, args.seed, args.backbone)
    if not args.dryrun and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"{output_dir} is not empty. Runs are not resumed: delete it to run again."
        )

    set_seed(args.seed)
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        task="taskB",
        use_90rotation=bool(MAIN["use_90rotation"]),
    )
    classes = list(train_loader.dataset.classes)
    model = create_model(args.backbone, num_classes=len(classes), pretrained=not args.dryrun)
    head, backbone = split_head_parameters(model, HEAD_SELECTOR)
    config = {
        "task": "Task-B",
        "step": "lr-selection",
        "backbone": args.backbone,
        "timm_name": BACKBONE_NAMES[args.backbone],
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "backbone_lr": args.backbone_lr,
        "head_lr": args.head_lr,
        "head_selector": HEAD_SELECTOR,
        "head_params": sum(p.numel() for p in head),
        "backbone_params": sum(p.numel() for p in backbone),
        "weight_decay": 1e-4,
        "label_smoothing": float(MAIN["label_smoothing"]),
        "use_90rotation": bool(MAIN["use_90rotation"]),
        "scheduler": "cosine",
        "warmup_epochs": args.warmup_epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "selection": "best val accuracy checkpoint; val/test scored from it",
        "amp": device == "cuda",
        "pretrained": "ImageNet",
        "data_root": relative(args.data_root),
        "split_sizes": {
            "train": len(train_loader.dataset),
            "val": len(val_loader.dataset),
            "test": len(test_loader.dataset),
        },
        "classes": classes,
        "git_commit": git_commit(),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "timm": timm.__version__,
        },
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
    }
    print(
        f"[{args.backbone} bblr={lr_tag(args.backbone_lr)}] head LR {args.head_lr} on "
        f"{config['head_params']:,} params, backbone LR {args.backbone_lr} on "
        f"{config['backbone_params']:,}; split {config['split_sizes']}"
    )
    if args.dryrun:
        images, _ = next(iter(val_loader))
        with torch.no_grad():
            logits = model.eval()(images[:2])
        assert logits.shape == (2, len(classes)), logits.shape
        print("DRYRUN PASSED: dataset, model, parameter groups and forward pass")
        return config

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "run_config.json", config)
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
    engine = TrainingEngine(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        use_amp=device == "cuda",
    )
    tracker = None
    if os.getenv("WANDB_API_KEY"):
        tracker = WandbTracker.start(
            project=args.wandb_project,
            name=f"{args.backbone}_bblr{lr_tag(args.backbone_lr)}_s{args.seed}",
            config=config,
            dir_path=str(output_dir),
        )
    try:
        result = engine.fit(
            epochs=args.epochs,
            output_dir=output_dir,
            name="model",
            save_best_only=True,
            early_stopping_patience=args.early_stopping_patience,
            tracker=tracker,
        )
    finally:
        if tracker:
            tracker.finish()
    write_json(
        output_dir / "history.json",
        [
            {"epoch": epoch + 1, **train, **val}
            for epoch, (train, val) in enumerate(zip(engine.train_history, engine.val_history))
        ],
    )

    best = torch.load(output_dir / "model_best.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    model.to(device)
    val, val_rows = score(model, val_loader, criterion, device)
    test, test_rows = score(model, test_loader, criterion, device)
    write_csv(output_dir / "predictions_val.csv", val_rows)
    write_csv(output_dir / "predictions_test.csv", test_rows)
    metrics = {
        **{key: config[key] for key in ("backbone", "seed", "backbone_lr", "head_lr", "head_selector")},
        "checkpoint": relative(output_dir / "model_best.pth"),
        "best_epoch": result["best_epoch"],
        "epochs_trained": result["epochs_trained"],
        "stopped_early": result["stopped_early"],
        "engine_best_val_acc": result["best_val_acc"],
        "val": val,
        "test": test,
    }
    write_json(output_dir / "metrics.json", metrics)  # completion marker, written last
    print(
        f"[{args.backbone} bblr={lr_tag(args.backbone_lr)}] best epoch {result['best_epoch']}: "
        f"val acc {val['accuracy'] * 100:.2f}% F1 {val['f1_macro'] * 100:.2f}% "
        f"loss {val['loss']:.4f} | test acc {test['accuracy'] * 100:.2f}%"
    )
    return metrics


if __name__ == "__main__":
    main()
