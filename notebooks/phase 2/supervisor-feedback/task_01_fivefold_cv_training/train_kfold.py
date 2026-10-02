#!/usr/bin/env python3
"""Origin CV with fixed epochs and fold-level restart; held-out evaluated once."""

import argparse
import gc
import os
from pathlib import Path
import sys
import uuid

import timm
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.factory import BACKBONE_NAMES
from src.train import WandbTracker
from src.utils import set_seed
from origin_cv import (
    PHASE1_MANIFEST,
    OriginKFoldDataset,
    SubsetWithTransform,
)
from run_artifacts import (
    PROTOCOL,
    completed_fold,
    git_commit,
    file_sha256,
    make_config,
    model_lock,
    prepare_run,
    process_token,
    relative_path,
    save_checkpoint,
    save_csv,
    save_json,
    utc_now,
)

# Match the main runs in phase 1/task-B-17-new/final_b_17class.py.
PHASE1_BATCH_SIZES = {
    "resnet50": 64,
    "efficientnet_b3": 32,
    "mobilenetv3_large": 64,
    "convnext_tiny": 32,
    "swin_tiny": 32,
}


def get_transforms(img_size=224, use_90rotation=False):
    """Reuse Phase 1 main augmentation; 90-degree rotation is an opt-in ablation."""
    return (
        get_train_transforms(img_size=img_size, use_90rotation=use_90rotation),
        get_val_transforms(img_size=img_size),
    )


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum = correct = total = 0
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(labels)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += len(labels)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate_full(model, loader, device):
    """Return predictions in dataset order; called once after fixed epochs."""
    model.eval()
    labels, predictions, confidence = [], [], []
    for images, targets in tqdm(loader, desc="Held-out", leave=False):
        probabilities = model(images.to(device)).softmax(1).cpu()
        score, predicted = probabilities.max(1)
        labels.extend(targets.tolist())
        predictions.extend(predicted.tolist())
        confidence.extend(score.tolist())
    return labels, predictions, confidence


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", required=True, choices=list(BACKBONE_NAMES))
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--fold", type=int, help="Run only this fold (1-based)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Override Phase 1 batch: ResNet/MobileNet 64; other models 32",
    )
    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--data_root", type=Path, default=ROOT / "data/task-B-group-split"
    )
    parser.add_argument(
        "--output_root", type=Path, default=ROOT / "data/experiments/Task-B-origin-cv"
    )
    parser.add_argument("--manifest_path", type=Path, default=PHASE1_MANIFEST)
    parser.add_argument("--wandb_project", default="woodbench-origin-cv")
    parser.add_argument("--use_90rotation", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep matching completed folds; restart unfinished folds from epoch 0",
    )
    parser.add_argument("--dryrun", action="store_true")
    return parser


def run_fold(args, dataset, split, config, fingerprint, output_dir, device, transforms):
    fold = split["fold"] + 1
    folder = output_dir / f"fold_{fold}"
    folder.mkdir(parents=True, exist_ok=True)
    for name in (
        "last.pth",
        "best.pth",
        "final.pth",
        "predictions.csv",
        "history.json",
    ):
        (folder / name).unlink(missing_ok=True)
    run_id = f"{args.backbone}-s{args.seed}-f{fold}-{uuid.uuid4().hex}"
    metadata = {
        "run_id": run_id,
        "run_date": utc_now(),
        "git_commit": git_commit(),
        "notebook": relative_path(Path(__file__).parent / "fivefold_cv_training.ipynb"),
        "entrypoint": relative_path(Path(__file__)),
        "fold": fold,
        "backbone": args.backbone,
        "fingerprint": fingerprint,
        "protocol": PROTOCOL,
        "config": config,
        "split_version": config["split_version"],
        "class_to_idx": dataset.class_to_idx,
        "num_classes": len(dataset.classes),
        "n_train": split["n_train"],
        "n_val": 0,
        "n_test": split["n_val"],
        "train_origins": sorted(split["train_origins"]),
        "test_origins": sorted(split["val_origins"]),
        "checkpoint_path": relative_path(folder / "final.pth"),
        "prediction_path": relative_path(folder / "predictions.csv"),
        "pid": os.getpid(),
        "process_token": process_token(os.getpid()),
        "status": "running",
    }
    save_json(folder / "run.json", metadata)
    tracker = None
    try:
        # Seed before constructing a fresh model, also when restarting a fold.
        set_seed(args.seed)
        train_data = SubsetWithTransform(dataset, split["train_idx"], transforms[0])
        test_data = SubsetWithTransform(dataset, split["val_idx"], transforms[1])
        loader_args = dict(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        train_loader = DataLoader(train_data, shuffle=True, **loader_args)
        test_loader = DataLoader(test_data, shuffle=False, **loader_args)
        model = timm.create_model(
            BACKBONE_NAMES[args.backbone],
            pretrained=True,
            num_classes=len(dataset.classes),
        ).to(device)
        head, backbone = [], []
        for name, parameter in model.named_parameters():
            if any(key in name for key in ("head", "fc", "classifier")):
                head.append(parameter)
            else:
                backbone.append(parameter)
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone, "lr": args.backbone_lr},
                {"params": head, "lr": args.head_lr},
            ],
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        if os.getenv("WANDB_API_KEY"):
            tracker = WandbTracker.start(
                project=args.wandb_project, name=run_id, config=config
            )
        history = []
        for epoch in range(args.epochs):
            loss, accuracy = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            scheduler.step()
            history.append(
                {"epoch": epoch + 1, "train_loss": loss, "train_accuracy": accuracy}
            )
            save_json(folder / "history.json", history)
            metadata["epoch"] = epoch + 1
            save_json(folder / "run.json", metadata)
            if tracker:
                tracker.log(history[-1], step=epoch + 1)
            print(f"Fold {fold}, epoch {epoch + 1}/{args.epochs}: train={accuracy:.4f}")
        save_checkpoint(
            folder / "final.pth",
            {
                "model_state_dict": model.state_dict(),
                "epoch": args.epochs - 1,
                "run_id": run_id,
                "config": config,
                "fingerprint": fingerprint,
            },
        )
        labels, predictions, confidences = evaluate_full(model, test_loader, device)
        rows = []
        for index, label, prediction, confidence in zip(
            split["val_idx"], labels, predictions, confidences
        ):
            record = dataset.records[index]
            rows.append(
                {
                    "run_id": run_id,
                    "fold": fold,
                    "path": relative_path(record["path"]),
                    "origin_key": record["origin_key"],
                    "md5": record["md5"],
                    "source_split": record["source_split"],
                    "true_label": label,
                    "predicted_label": prediction,
                    "true_species": dataset.classes[label],
                    "predicted_species": dataset.classes[prediction],
                    "confidence": confidence,
                }
            )
        if len(rows) != split["n_val"]:
            raise ValueError("Held-out prediction count does not match split")
        save_csv(folder / "predictions.csv", rows)
        # Full per-class/OOF/report and registry export is in evaluate_and_register_cv.ipynb.
        metadata.update(
            status="completed",
            finished_at=utc_now(),
            accuracy=sum(
                label == prediction for label, prediction in zip(labels, predictions)
            )
            / len(labels),
            artifact_sha256={
                name: file_sha256(folder / name)
                for name in ("final.pth", "predictions.csv")
            },
        )
        save_json(folder / "metrics.json", metadata)  # completion marker written last
        save_json(folder / "run.json", metadata)
        if tracker:
            tracker.log(
                {"held_out_accuracy": metadata["accuracy"]}, step=args.epochs + 1
            )
        return metadata
    except BaseException:
        metadata.update(status="interrupted", finished_at=utc_now())
        save_json(folder / "run.json", metadata)
        raise
    finally:
        if tracker:
            tracker.finish()
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.batch_size is None:
        args.batch_size = PHASE1_BATCH_SIZES[args.backbone]
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError(
            "epochs/batch_size must be positive; num_workers must be nonnegative"
        )
    if args.fold is not None and not 1 <= args.fold <= args.n_folds:
        raise ValueError(f"Fold must be between 1 and {args.n_folds}")
    dataset = OriginKFoldDataset(
        args.data_root, "all", manifest_path=args.manifest_path
    )
    splits = dataset.get_splits_by_origin(args.n_folds, seed=args.seed)
    config = make_config(args, dataset, splits)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; dataset: {len(dataset)}; classes: {len(dataset.classes)}")
    print(f"Training: {args.epochs} epochs/fold, batch={args.batch_size}")
    for split in splits:
        print(
            f"Fold {split['fold'] + 1}: {split['n_train']} train, {split['n_val']} held-out"
        )
    if args.dryrun:
        model = timm.create_model(
            BACKBONE_NAMES[args.backbone],
            pretrained=False,
            num_classes=len(dataset.classes),
        )
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            save_checkpoint(
                path / "model.pth",
                {"model_state_dict": model.state_dict(), "config": config},
            )
            torch.load(path / "model.pth", map_location="cpu", weights_only=True)
            save_json(path / "config.json", config)
        print("DRYRUN PASSED: origin folds, model, JSON and checkpoint round-trip")
        return
    output_dir = args.output_root / args.backbone
    with model_lock(output_dir):
        fingerprint = prepare_run(output_dir, config, args.resume)
        transforms = get_transforms(use_90rotation=args.use_90rotation)
        for split in splits:
            if args.fold is not None and split["fold"] + 1 != args.fold:
                continue
            folder = output_dir / f"fold_{split['fold'] + 1}"
            completed = completed_fold(folder, fingerprint)
            if completed:
                print(
                    f"Keeping completed fold {split['fold'] + 1}: {completed['run_id']}"
                )
                continue
            run_fold(
                args,
                dataset,
                split,
                config,
                fingerprint,
                output_dir,
                device,
                transforms,
            )
        results = [
            completed_fold(output_dir / f"fold_{i + 1}", fingerprint)
            for i in range(args.n_folds)
        ]
        results = [result for result in results if result is not None]
        summary = {
            "config": config,
            "fingerprint": fingerprint,
            "n_completed": len(results),
            "n_folds": args.n_folds,
            "source_run_ids": [r["run_id"] for r in results],
        }
        save_json(output_dir / "summary.json", summary)
        print(
            f"Completed {len(results)}/{args.n_folds} folds. Run evaluate_and_register_cv.ipynb for reports."
        )
        return summary


if __name__ == "__main__":
    main()
