"""
Fine-tuning utilities for 20-class → 17-class adaptation.

Loads pretrained 20-class checkpoints, transfers backbone weights to a new
17-class model, then fine-tunes in 2 stages:
  Stage 1: Freeze backbone, train classifier head only (5 epochs, LR=1e-3)
  Stage 2: Unfreeze all, fine-tune end-to-end (15 epochs, backbone_lr=1e-5)

Usage (from notebook):
    from finetune_17class import run_all_experiments
    results = run_all_experiments('resnet50', ROOT, device='cuda')

Requires ROOT on sys.path so that `from src import ...` works.
"""

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.models.factory import create_model
from src.data.transforms import get_train_transforms, get_val_transforms
from src.eval.metrics import compute_metrics
from src.utils.seed import set_seed


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

CLASSES_TO_REMOVE = [
    "Calocedrus sp",
    "Cupressus funebris (Cupressus pendula)",
    "Pinus sp",
]

BACKBONES = [
    "resnet50",
    "efficientnet_b3",
    "mobilenetv3_large",
    "convnext_tiny",
    "swin_tiny",
]

# Experiment configs: main uses 3 seeds, ablations use seed_42 only
EXPERIMENTS = {
    "main": {
        "seeds": [42, 123, 456],
        "label_smoothing": 0.1,
        "use_90rotation": False,
        "ckpt_pattern": "data/experiments/Task-B/main/seed_{seed}/{backbone}/{backbone}_seed{seed}_best.pth",
    },
    "taskB_90rotation": {
        "seeds": [42],
        "label_smoothing": 0.1,
        "use_90rotation": True,
        "ckpt_pattern": "data/experiments/Task-B/taskB_90rotation/{backbone}/{backbone}_seed42_best.pth",
    },
    "taskB_label_smoothing_0.0": {
        "seeds": [42],
        "label_smoothing": 0.0,
        "use_90rotation": False,
        "ckpt_pattern": "data/experiments/Task-B/taskB_label_smoothing_0.0/{backbone}/{backbone}_seed42_best.pth",
    },
    "taskB_label_smoothing_0.2": {
        "seeds": [42],
        "label_smoothing": 0.2,
        "use_90rotation": False,
        "ckpt_pattern": "data/experiments/Task-B/taskB_label_smoothing_0.2/{backbone}/{backbone}_seed42_best.pth",
    },
}

# ── Hyperparameters ────────────────────────────────────────────────────────
STAGE1_EPOCHS = 5
STAGE1_LR = 1e-3
STAGE2_EPOCHS = 15
STAGE2_LR = 1e-5
STAGE2_HEAD_LR_MULT = 10  # Head gets 10× the backbone LR in stage 2
BATCH_SIZE = 64
NUM_WORKERS = 4
WEIGHT_DECAY = 1e-4
IMG_SIZE = 224


# ═══════════════════════════════════════════════════════════════════════════
# Data preparation
# ═══════════════════════════════════════════════════════════════════════════

def remove_excluded_classes(data_root: Path, dry_run: bool = False) -> dict:
    """Remove the 3 excluded class folders from all splits.

    Args:
        data_root: Path to dataset root (e.g. data/task-B-group-split)
        dry_run: If True, only report what would be deleted

    Returns:
        Dict mapping deleted paths to number of files removed
    """
    deleted = {}
    for split in ["train", "val", "test"]:
        split_dir = data_root / split
        if not split_dir.exists():
            continue
        for cls_name in CLASSES_TO_REMOVE:
            cls_dir = split_dir / cls_name
            if cls_dir.exists():
                n_files = len(list(cls_dir.glob("*")))
                if dry_run:
                    print(f"  [DRY RUN] Would delete: {cls_dir} ({n_files} files)")
                else:
                    shutil.rmtree(cls_dir)
                    print(f"  Deleted: {cls_dir} ({n_files} files)")
                deleted[str(cls_dir)] = n_files
    return deleted


def verify_dataset(data_root: Path, expected_classes: int = 17):
    """Verify dataset has exactly expected_classes in all splits."""
    print(f"\n{'─'*60}")
    print(f"Dataset verification: {data_root}")
    print(f"{'─'*60}")
    all_ok = True
    for split in ["train", "val", "test"]:
        split_dir = data_root / split
        classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
        n_images = sum(len(list((split_dir / c).glob("*"))) for c in classes)
        status = "✅" if len(classes) == expected_classes else "❌"
        print(f"  {split:5s}: {len(classes)} classes, {n_images} images {status}")
        if len(classes) != expected_classes:
            all_ok = False
            # Show which classes are present
            for c in CLASSES_TO_REMOVE:
                if c in classes:
                    print(f"         ⚠️  Still contains: {c}")
    assert all_ok, f"Expected {expected_classes} classes in all splits!"
    print(f"  All splits verified ✅")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Weight transfer & model utilities
# ═══════════════════════════════════════════════════════════════════════════

def transfer_backbone_weights(
    backbone_name: str,
    checkpoint_path: Path,
    num_classes_new: int = 17,
) -> nn.Module:
    """Load 20-class checkpoint and transfer backbone weights to a new model.

    Creates a fresh model with num_classes_new outputs, then copies all
    matching weights from the checkpoint (skipping the classifier head).
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_20 = ckpt["model_state_dict"]

    model = create_model(backbone_name, num_classes=num_classes_new, pretrained=False)
    state_new = model.state_dict()

    transferred, skipped = 0, 0
    skipped_keys = []
    for k in state_new:
        if k in state_20 and state_20[k].shape == state_new[k].shape:
            state_new[k] = state_20[k]
            transferred += 1
        else:
            skipped += 1
            skipped_keys.append(k)

    model.load_state_dict(state_new)
    print(f"  Transferred {transferred} params, skipped {skipped} (head: {skipped_keys})")
    return model


def freeze_backbone(model: nn.Module, num_classes: int = 17):
    """Freeze all parameters except the final classifier head."""
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze classifier — find Linear with out_features == num_classes
    for module in model.modules():
        if isinstance(module, nn.Linear) and module.out_features == num_classes:
            for param in module.parameters():
                param.requires_grad = True
            break
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Stage 1 — Frozen: {total - trainable:,} | Trainable: {trainable:,}")


def unfreeze_all(model: nn.Module):
    """Unfreeze all parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


def _get_param_groups(model: nn.Module, num_classes: int, backbone_lr: float, head_lr: float):
    """Split model params into backbone and head groups with different LRs."""
    head_params, backbone_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Detect classifier head by output dimension
        is_head = False
        if param.dim() == 2 and param.shape[0] == num_classes:
            is_head = True
        elif param.dim() == 1 and param.shape[0] == num_classes:
            is_head = True
        if is_head:
            head_params.append(param)
        else:
            backbone_params.append(param)
    return [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr},
    ]


# ═══════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════

def build_dataloaders(
    data_root: Path,
    use_90rotation: bool = False,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
):
    """Build train/val/test DataLoaders from ImageFolder structure."""
    train_tf = get_train_transforms(
        img_size=IMG_SIZE,
        use_90rotation=use_90rotation,
        degrees=30,
        hflip_prob=0.5,
        vflip_prob=0.5,
        brightness=0.2,
        contrast=0.2,
    )
    val_tf = get_val_transforms(img_size=IMG_SIZE)

    train_ds = ImageFolder(str(data_root / "train"), transform=train_tf)
    val_ds = ImageFolder(str(data_root / "val"), transform=val_tf)
    test_ds = ImageFolder(str(data_root / "test"), transform=val_tf)

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    print(f"  Data: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")
    print(f"  Classes ({len(train_ds.classes)}): {train_ds.classes[:3]}...{train_ds.classes[-1]}")
    return train_loader, val_loader, test_loader


def _train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=scaler is not None):
            outputs = model(images)
            loss = criterion(outputs, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return {"train_loss": total_loss / total, "train_acc": 100.0 * correct / total}


@torch.no_grad()
def _validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return {"val_loss": total_loss / total, "val_acc": 100.0 * correct / total}


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def finetune_from_checkpoint(
    backbone_name: str,
    checkpoint_path: Path,
    data_root: Path,
    output_dir: Path,
    seed: int,
    label_smoothing: float = 0.1,
    use_90rotation: bool = False,
    device: str = "cuda",
) -> dict:
    """Full fine-tune pipeline for a single run.

    Returns:
        Dict with test metrics and run metadata.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {backbone_name} | seed={seed} | ls={label_smoothing} | rot90={use_90rotation}")
    print(f"  ckpt: {checkpoint_path}")
    print(f"  out:  {output_dir}")
    print(f"{'='*60}")

    set_seed(seed)

    # 1. Data
    train_loader, val_loader, test_loader = build_dataloaders(
        data_root, use_90rotation=use_90rotation,
    )
    num_classes = len(train_loader.dataset.classes)
    class_names = list(train_loader.dataset.classes)

    # 2. Transfer backbone weights
    print(f"\n[1/4] Transferring backbone weights...")
    model = transfer_backbone_weights(backbone_name, checkpoint_path, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = GradScaler() if device == "cuda" else None
    best_val_acc = 0.0
    best_state = None

    # 3. Stage 1: Freeze backbone, train head
    print(f"\n[2/4] Stage 1: Linear probing ({STAGE1_EPOCHS} ep, LR={STAGE1_LR})")
    freeze_backbone(model, num_classes)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=STAGE1_LR, weight_decay=WEIGHT_DECAY,
    )
    for epoch in range(STAGE1_EPOCHS):
        tm = _train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        vm = _validate(model, val_loader, criterion, device)
        if vm["val_acc"] > best_val_acc:
            best_val_acc = vm["val_acc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"  E{epoch+1}/{STAGE1_EPOCHS} | "
            f"train {tm['train_loss']:.4f} / {tm['train_acc']:.1f}% | "
            f"val {vm['val_loss']:.4f} / {vm['val_acc']:.1f}%"
        )

    # 4. Stage 2: Unfreeze all, fine-tune with differential LR
    print(f"\n[3/4] Stage 2: Full fine-tune ({STAGE2_EPOCHS} ep, backbone_lr={STAGE2_LR})")
    unfreeze_all(model)
    param_groups = _get_param_groups(
        model, num_classes,
        backbone_lr=STAGE2_LR,
        head_lr=STAGE2_LR * STAGE2_HEAD_LR_MULT,
    )
    optimizer = AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=STAGE2_EPOCHS)

    for epoch in range(STAGE2_EPOCHS):
        tm = _train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        vm = _validate(model, val_loader, criterion, device)
        scheduler.step()
        if vm["val_acc"] > best_val_acc:
            best_val_acc = vm["val_acc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  E{epoch+1}/{STAGE2_EPOCHS} | "
                f"train {tm['train_loss']:.4f} / {tm['train_acc']:.1f}% | "
                f"val {vm['val_loss']:.4f} / {vm['val_acc']:.1f}%"
            )

    # 5. Evaluate best model on test set
    print(f"\n[4/4] Test evaluation (best val_acc={best_val_acc:.1f}%)...")
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    test_metrics = compute_metrics(
        model=model,
        data_loader=test_loader,
        device=device,
        num_classes=num_classes,
        class_names=class_names,
    )

    # 6. Save checkpoint + metrics
    ckpt_save = output_dir / f"{backbone_name}_seed{seed}_best.pth"
    torch.save(
        {
            "epoch": STAGE1_EPOCHS + STAGE2_EPOCHS,
            "model_state_dict": model.state_dict(),
            "best_val_acc": best_val_acc,
            "num_classes": num_classes,
            "class_names": class_names,
        },
        ckpt_save,
    )

    result = {
        "backbone": backbone_name,
        "seed": seed,
        "best_val_acc": best_val_acc,
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_f1_macro": float(test_metrics["f1_macro"]),
        "test_precision_macro": float(test_metrics["precision_macro"]),
        "test_recall_macro": float(test_metrics["recall_macro"]),
    }
    if "per_class" in test_metrics:
        result["per_class"] = {
            cls: {k: float(v) for k, v in scores.items()}
            for cls, scores in test_metrics["per_class"].items()
        }
    # Save confusion matrix as numpy
    if "confusion_matrix" in test_metrics:
        cm_path = output_dir / "confusion_matrix.npy"
        np.save(cm_path, test_metrics["confusion_matrix"])

    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(
        f"  ✅ Acc={test_metrics['accuracy']*100:.2f}% | "
        f"F1={test_metrics['f1_macro']*100:.2f}% | "
        f"Saved → {ckpt_save.name}"
    )
    return result


def run_all_experiments(
    backbone_name: str,
    root: Path,
    data_root: Path | None = None,
    output_root: Path | None = None,
    device: str = "cuda",
) -> list[dict]:
    """Run all experiment configs for a single backbone.

    main: 3 seeds (42, 123, 456)
    ablations: seed 42 only

    Args:
        backbone_name: One of BACKBONES
        root: Project root (ROOT)
        data_root: 17-class data dir (default: ROOT/data/task-B-group-split)
        output_root: Output dir (default: ROOT/data/experiments/Task-B-17class)
        device: 'cuda' or 'cpu'

    Returns:
        List of result dicts for all runs
    """
    root = Path(root)
    data_root = Path(data_root) if data_root else root / "data" / "task-B-group-split"
    output_root = Path(output_root) if output_root else root / "data" / "experiments" / "Task-B-17class"

    all_results = []
    t_start = time.time()
    total_runs = sum(len(cfg["seeds"]) for cfg in EXPERIMENTS.values())

    print(f"\n{'═'*60}")
    print(f"  {backbone_name} — {total_runs} runs across {len(EXPERIMENTS)} configs")
    print(f"{'═'*60}")

    run_idx = 0
    for exp_name, exp_cfg in EXPERIMENTS.items():
        for seed in exp_cfg["seeds"]:
            run_idx += 1
            ckpt_path = root / exp_cfg["ckpt_pattern"].format(
                backbone=backbone_name, seed=seed,
            )
            if not ckpt_path.exists():
                print(f"\n⚠️  [{run_idx}/{total_runs}] Checkpoint missing: {ckpt_path}")
                continue

            # Output directory
            if exp_name == "main":
                out_dir = output_root / "main" / f"seed_{seed}" / backbone_name
            else:
                out_dir = output_root / exp_name / backbone_name

            print(f"\n▶ [{run_idx}/{total_runs}] {exp_name} / seed={seed}")
            result = finetune_from_checkpoint(
                backbone_name=backbone_name,
                checkpoint_path=ckpt_path,
                data_root=data_root,
                output_dir=out_dir,
                seed=seed,
                label_smoothing=exp_cfg["label_smoothing"],
                use_90rotation=exp_cfg["use_90rotation"],
                device=device,
            )
            result["experiment"] = exp_name
            all_results.append(result)

    elapsed = time.time() - t_start
    print(f"\n{'═'*60}")
    print(f"  {backbone_name} — {len(all_results)}/{total_runs} runs done in {elapsed/60:.1f} min")
    print(f"{'═'*60}")

    # Summary table
    print(f"\n{'─'*60}")
    print(f"  {'Experiment':<30s} {'Seed':>4s} {'Acc%':>6s} {'F1%':>6s}")
    print(f"  {'─'*50}")
    for r in all_results:
        print(
            f"  {r['experiment']:<30s} {r['seed']:>4d} "
            f"{r['test_accuracy']*100:>6.2f} {r['test_f1_macro']*100:>6.2f}"
        )
    print(f"{'─'*60}")

    # Save aggregated results
    agg_path = output_root / f"{backbone_name}_all_results.json"
    with open(agg_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"  Aggregated results → {agg_path}")

    return all_results
