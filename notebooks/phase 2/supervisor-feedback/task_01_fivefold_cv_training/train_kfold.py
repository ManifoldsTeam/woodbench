#!/usr/bin/env python3
"""Origin CV with inner origin validation, best-val checkpoint and fold restart.

Each fold trains on its outer-train origins minus an inner validation set of
whole origins, keeps the checkpoint with the best inner-val accuracy (early
stopping as in Phase 1 Final-B) and evaluates the outer held-out fold once.
"""

import argparse
import gc
import os
from pathlib import Path
import sys
import uuid

import timm
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.factory import BACKBONE_NAMES
from src.train import WandbTracker, create_optimizer, create_scheduler
from src.train.optimizer import split_head_parameters
from src.utils import set_seed
from origin_cv import (
    PHASE1_MANIFEST,
    OriginKFoldDataset,
    SubsetWithTransform,
)
from run_artifacts import (
    CHECKPOINT,
    INPUT_VARIANTS,
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


def degrade_input(input_variant, img_size=224):
    """Input degradation applied before augmentation to every split."""
    if input_variant == "rgb":
        return []
    if input_variant == "grayscale":
        return [T.Grayscale(num_output_channels=3)]
    if input_variant == "px16":
        # Same as the July anatomy probe: resize 224 -> 16 -> 224, bilinear.
        return [
            T.Resize((img_size, img_size)),
            T.Resize((16, 16)),
            T.Resize((img_size, img_size)),
        ]
    raise ValueError(f"Unknown input_variant: {input_variant}")


def get_transforms(img_size=224, use_90rotation=False, input_variant="rgb"):
    """Reuse Phase 1 main augmentation; 90-degree rotation is an opt-in ablation."""
    train = get_train_transforms(img_size=img_size, use_90rotation=use_90rotation)
    held_out = get_val_transforms(img_size=img_size)
    degrade = degrade_input(input_variant, img_size)
    if not degrade:
        return train, held_out
    return T.Compose(degrade + [train]), T.Compose(degrade + [held_out])


def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """One epoch; with a GradScaler, AMP exactly as src/train/trainer.py."""
    model.train()
    loss_sum = correct = total = 0
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.autocast(device.type, enabled=scaler is not None):
            outputs = model(images)
            loss = criterion(outputs, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        loss_sum += loss.item() * len(labels)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += len(labels)
    return loss_sum / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp=False):
    """Inner-validation loss/accuracy, computed like TrainingEngine.validate."""
    model.eval()
    loss_sum = correct = total = 0
    for images, labels in tqdm(loader, desc="Val", leave=False):
        images, labels = images.to(device), labels.to(device)
        with torch.autocast(device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        loss_sum += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += len(labels)
    return loss_sum / len(loader), correct / total


def class_metrics(labels, predictions, classes):
    """Macro and per-species precision/recall/F1 for one fold."""
    indices = list(range(len(classes)))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=indices, zero_division=0
    )
    return {
        "f1_macro": float(
            f1_score(labels, predictions, labels=indices, average="macro", zero_division=0)
        ),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(classes)
        },
    }


@torch.no_grad()
def evaluate_full(model, loader, device):
    """Return predictions in dataset order; called once with the best checkpoint."""
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
    parser.add_argument(
        "--backbone_lr",
        type=float,
        required=True,
        help="Chosen in step 1 (task_07_lr_selection); no default on purpose",
    )
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="Epochs without inner-val accuracy gain; 0 disables",
    )
    parser.add_argument(
        "--inner_val_fraction",
        type=float,
        default=0.2,
        help="Share of each species' outer-train origins used for validation",
    )
    parser.add_argument(
        "--input_variant", choices=list(INPUT_VARIANTS), default="rgb"
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--data_root", type=Path, default=ROOT / "data/task-B-group-split"
    )
    parser.add_argument(
        "--output_root", type=Path, default=ROOT / "data/experiments/Task-B-origin-cv-v2"
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
        "n_train": len(split["inner_train_idx"]),
        "n_val": len(split["inner_val_idx"]),
        "n_test": split["n_val"],
        "train_origins": sorted(split["inner_train_origins"]),
        "val_origins": sorted(split["inner_val_origins"]),
        "test_origins": sorted(split["val_origins"]),
        "checkpoint_path": relative_path(folder / CHECKPOINT),
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
        train_data = SubsetWithTransform(dataset, split["inner_train_idx"], transforms[0])
        val_data = SubsetWithTransform(dataset, split["inner_val_idx"], transforms[1])
        test_data = SubsetWithTransform(dataset, split["val_idx"], transforms[1])
        loader_args = dict(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        train_loader = DataLoader(train_data, shuffle=True, **loader_args)
        val_loader = DataLoader(val_data, shuffle=False, **loader_args)
        test_loader = DataLoader(test_data, shuffle=False, **loader_args)
        model = timm.create_model(
            BACKBONE_NAMES[args.backbone],
            pretrained=True,
            num_classes=len(dataset.classes),
        ).to(device)
        head, backbone = split_head_parameters(model, config["head_selector"])
        metadata["parameter_groups"] = {
            "head": sum(p.numel() for p in head),
            "backbone": sum(p.numel() for p in backbone),
        }
        optimizer = create_optimizer(
            model,
            lr=args.head_lr,
            weight_decay=args.weight_decay,
            backbone_lr=args.backbone_lr,
            head_selector=config["head_selector"],
        )
        scheduler = create_scheduler(
            optimizer,
            epochs=args.epochs,
            scheduler_type="cosine",
            warmup_epochs=args.warmup_epochs,
        )
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        use_amp = device.type == "cuda"
        scaler = torch.amp.GradScaler(device.type) if use_amp else None
        if os.getenv("WANDB_API_KEY"):
            tracker = WandbTracker.start(
                project=args.wandb_project, name=run_id, config=config
            )
        history = []
        best = {"val_accuracy": -1.0}
        stale = 0
        for epoch in range(args.epochs):
            loss, accuracy = train_epoch(
                model, train_loader, criterion, optimizer, device, scaler
            )
            val_loss, val_accuracy = validate(
                model, val_loader, criterion, device, use_amp
            )
            lr = {group.get("name", "all"): group["lr"] for group in optimizer.param_groups}
            scheduler.step()
            improved = val_accuracy > best["val_accuracy"]
            if improved:
                best = {
                    "epoch": epoch + 1,
                    "val_accuracy": val_accuracy,
                    "val_loss": val_loss,
                }
                stale = 0
                save_checkpoint(
                    folder / CHECKPOINT,
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "run_id": run_id,
                        "config": config,
                        "fingerprint": fingerprint,
                        "val_accuracy": val_accuracy,
                        "val_loss": val_loss,
                    },
                )
            else:
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
                }
            )
            save_json(folder / "history.json", history)
            metadata["epoch"] = epoch + 1
            save_json(folder / "run.json", metadata)
            if tracker:
                tracker.log(history[-1], step=epoch + 1)
            print(
                f"Fold {fold}, epoch {epoch + 1}/{args.epochs}: train={accuracy:.4f} "
                f"val={val_accuracy:.4f} loss={val_loss:.4f}{' [best]' if improved else ''}"
            )
            if args.early_stopping_patience and stale >= args.early_stopping_patience:
                print(f"Early stopping after {args.early_stopping_patience} epochs without gain")
                break
        checkpoint = torch.load(folder / CHECKPOINT, map_location="cpu", weights_only=True)
        if checkpoint["run_id"] != run_id:
            raise ValueError("Best checkpoint belongs to another run")
        model.load_state_dict(checkpoint["model_state_dict"])
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
        # Registry, OOF and per-species fold summaries are in evaluate_and_register_cv.ipynb.
        metadata.update(
            status="completed",
            finished_at=utc_now(),
            best_epoch=best["epoch"],
            best_val_accuracy=best["val_accuracy"],
            best_val_loss=best["val_loss"],
            epochs_trained=len(history),
            stopped_early=len(history) < args.epochs,
            accuracy=sum(
                label == prediction for label, prediction in zip(labels, predictions)
            )
            / len(labels),
            **class_metrics(labels, predictions, dataset.classes),
            artifact_sha256={
                name: file_sha256(folder / name)
                for name in (CHECKPOINT, "predictions.csv")
            },
        )
        save_json(folder / "metrics.json", metadata)  # completion marker written last
        save_json(folder / "run.json", metadata)
        if tracker:
            tracker.log(
                {"held_out_accuracy": metadata["accuracy"]}, step=len(history) + 1
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
    if not 0 <= args.warmup_epochs < args.epochs or args.early_stopping_patience < 0:
        raise ValueError(
            "warmup_epochs must be in [0, epochs); early_stopping_patience nonnegative"
        )
    if args.fold is not None and not 1 <= args.fold <= args.n_folds:
        raise ValueError(f"Fold must be between 1 and {args.n_folds}")
    dataset = OriginKFoldDataset(
        args.data_root, "all", manifest_path=args.manifest_path
    )
    splits = dataset.get_splits_by_origin(
        args.n_folds, seed=args.seed, inner_val_fraction=args.inner_val_fraction
    )
    config = make_config(args, dataset, splits)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; dataset: {len(dataset)}; classes: {len(dataset.classes)}")
    print(
        f"Training: up to {args.epochs} epochs/fold, batch={args.batch_size}, "
        f"backbone_lr={args.backbone_lr}, head_lr={args.head_lr}, "
        f"input={args.input_variant}"
    )
    for split in splits:
        print(
            f"Fold {split['fold'] + 1}: {len(split['inner_train_idx'])} train, "
            f"{len(split['inner_val_idx'])} val, {split['n_val']} held-out"
        )
    if args.dryrun:
        model = timm.create_model(
            BACKBONE_NAMES[args.backbone],
            pretrained=False,
            num_classes=len(dataset.classes),
        )
        head, backbone = split_head_parameters(model, config["head_selector"])
        print(
            f"Head LR {args.head_lr} on {sum(p.numel() for p in head):,} params; "
            f"backbone LR {args.backbone_lr} on {sum(p.numel() for p in backbone):,}"
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
        print("DRYRUN PASSED: origin folds, inner val, model, JSON and checkpoint round-trip")
        return
    output_dir = args.output_root / args.backbone
    with model_lock(output_dir):
        fingerprint = prepare_run(output_dir, config, args.resume)
        transforms = get_transforms(
            use_90rotation=args.use_90rotation, input_variant=args.input_variant
        )
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
