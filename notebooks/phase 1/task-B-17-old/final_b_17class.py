"""Shared training utilities for final Task B 17-class notebooks.

These notebooks train Task B directly on species labels from
data/task-B-group-split after the 3 excluded classes have been removed.
Models start from ImageNet pretrained weights, not from the old 20-class
Task B checkpoints.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.data import get_dataloaders
from src.models import create_model
from src.train import (
    TrainingEngine,
    WandbTracker,
    create_loss_function,
    create_optimizer,
    create_scheduler,
)
from src.utils import set_seed

EXCLUDED_CLASSES = [
    "Calocedrus sp",
    "Cupressus funebris (Cupressus pendula)",
    "Pinus sp",
]

BACKBONE_CONFIGS: dict[str, dict[str, Any]] = {
    "resnet50": {"display_name": "ResNet-50", "batch_size": 64},
    "efficientnet_b3": {"display_name": "EfficientNet-B3", "batch_size": 32},
    "mobilenetv3_large": {"display_name": "MobileNetV3-Large", "batch_size": 64},
    "convnext_tiny": {"display_name": "ConvNeXt-Tiny", "batch_size": 32},
    "swin_tiny": {"display_name": "Swin-Tiny", "batch_size": 32},
}

EXPERIMENTS: dict[str, dict[str, Any]] = {
    "main": {
        "seeds": [42, 123, 456],
        "label_smoothing": 0.1,
        "use_90rotation": False,
        "tags": ["main"],
    },
    "taskB_90rotation": {
        "seeds": [42],
        "label_smoothing": 0.1,
        "use_90rotation": True,
        "tags": ["ablation", "90rotation"],
    },
    "taskB_label_smoothing_0.0": {
        "seeds": [42],
        "label_smoothing": 0.0,
        "use_90rotation": False,
        "tags": ["ablation", "label_smoothing_0.0"],
    },
    "taskB_label_smoothing_0.2": {
        "seeds": [42],
        "label_smoothing": 0.2,
        "use_90rotation": False,
        "tags": ["ablation", "label_smoothing_0.2"],
    },
}

OUTPUT_EXPERIMENT_ROOT = Path("data") / "experiments" / "Task-B-17class"


def prepare_environment(root: Path) -> None:
    """Load optional env vars and make WandB non-interactive without a key."""
    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env")
    except ImportError:
        pass

    os.environ.setdefault("WANDB_API_KEY", os.getenv("WANDB_API_KEY", ""))
    os.environ.setdefault("HF_TOKEN", os.getenv("HF_TOKEN", ""))
    if not os.getenv("WANDB_API_KEY"):
        os.environ.setdefault("WANDB_MODE", "disabled")


def verify_17class_dataset(data_root: Path, expected_classes: int = 17) -> list[str]:
    """Verify flat Task B split folders contain exactly the final 17 species."""
    data_root = Path(data_root)
    split_classes: dict[str, list[str]] = {}
    errors: list[str] = []

    for split in ["train", "val", "test"]:
        split_dir = data_root / split
        if not split_dir.exists():
            errors.append(f"{split}/ not found under {data_root}")
            split_classes[split] = []
            continue

        classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
        split_classes[split] = classes
        present_excluded = [name for name in EXCLUDED_CLASSES if name in classes]
        if len(classes) != expected_classes:
            errors.append(
                f"{split}: expected {expected_classes} classes, found {len(classes)}"
            )
        if present_excluded:
            errors.append(
                f"{split}: excluded classes still present: {present_excluded}"
            )

    reference = split_classes.get("train", [])
    for split in ["val", "test"]:
        if split_classes.get(split, []) != reference:
            missing = sorted(set(reference) - set(split_classes.get(split, [])))
            extra = sorted(set(split_classes.get(split, [])) - set(reference))
            errors.append(
                f"{split}: class mismatch vs train; missing={missing}, extra={extra}"
            )

    if errors:
        details = "\n".join(f"  - {msg}" for msg in errors)
        raise ValueError(
            "Task B final notebooks require the 17-class dataset.\n"
            "Run notebooks/phase 1/task-B-17-old/00_data_preparation.ipynb first, "
            "or use the already-prepared cloud data.\n"
            f"{details}"
        )

    for split, classes in split_classes.items():
        n_images = sum(
            len(list((data_root / split / cls).glob("*.jpg")))
            + len(list((data_root / split / cls).glob("*.jpeg")))
            + len(list((data_root / split / cls).glob("*.png")))
            for cls in classes
        )
        print(f"[Data] {split}: {len(classes)} classes, {n_images} images")

    return reference


def experiment_output_dir(
    root: Path, experiment_name: str, seed: int, backbone: str
) -> Path:
    """Output layout matching the Task B notebook convention."""
    base = root / OUTPUT_EXPERIMENT_ROOT / experiment_name
    if experiment_name == "main":
        return base / f"seed_{seed}" / backbone
    return base / backbone


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _checkpoint_result(ckpt: dict[str, Any]) -> dict[str, Any]:
    metrics = ckpt.get("metrics", {})
    best_val_acc = float(ckpt.get("best_val_acc", metrics.get("val_acc", np.nan)))
    best_epoch = int(ckpt.get("best_epoch", ckpt.get("epoch", 0))) + 1
    epochs_trained = int(ckpt.get("epoch", best_epoch - 1)) + 1
    return {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "final_train_acc": float(metrics.get("train_acc", np.nan)),
        "final_val_acc": float(metrics.get("val_acc", best_val_acc)),
        "stopped_early": False,
        "epochs_trained": epochs_trained,
    }


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, test_loader, device: str) -> dict[str, Any]:
    """Evaluate a model and return aggregate plus per-class test metrics."""
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    class_names = list(test_loader.dataset.classes)

    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    return {
        "test_accuracy": float(accuracy_score(all_labels, all_preds)),
        "test_f1_macro": float(
            f1_score(all_labels, all_preds, average="macro", zero_division=0)
        ),
        "test_precision_macro": float(
            precision_score(all_labels, all_preds, average="macro", zero_division=0)
        ),
        "test_recall_macro": float(
            recall_score(all_labels, all_preds, average="macro", zero_division=0)
        ),
        "test_f1_per_class": {
            class_name: float(score)
            for class_name, score in zip(class_names, f1_per_class)
        },
    }


def _write_metrics(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_completed_results(
    *, backbone_name: str, root: Path
) -> list[dict[str, Any]]:
    """Load completed metrics for one backbone in experiment order."""
    results: list[dict[str, Any]] = []
    for experiment_name, experiment_cfg in EXPERIMENTS.items():
        for seed in experiment_cfg["seeds"]:
            output_dir = experiment_output_dir(
                root, experiment_name, seed, backbone_name
            )
            metrics_path = output_dir / "test_metrics.json"
            if metrics_path.exists():
                results.append(
                    json.loads(metrics_path.read_text(encoding="utf-8"))
                )
    return results


def _evaluate_existing_checkpoint(
    *,
    backbone: str,
    seed: int,
    experiment_name: str,
    experiment_cfg: dict[str, Any],
    data_root: Path,
    output_dir: Path,
    batch_size: int,
    device: str,
) -> dict[str, Any] | None:
    ckpt_path = output_dir / f"{backbone}_seed{seed}_best.pth"
    if not ckpt_path.exists():
        return None

    print(f"[{experiment_name} | {backbone} s{seed}] Found checkpoint: {ckpt_path}")
    _, _, test_loader = get_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=4,
        task="taskB",
        use_90rotation=bool(experiment_cfg["use_90rotation"]),
    )
    num_classes = len(test_loader.dataset.classes)

    model = create_model(backbone, num_classes=num_classes, pretrained=False)
    ckpt = _load_checkpoint(ckpt_path)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    result = _checkpoint_result(ckpt)
    test_metrics = evaluate_model(model, test_loader, device)
    payload = {
        "backbone": backbone,
        "seed": seed,
        "experiment": experiment_name,
        "num_classes": num_classes,
        **result,
        **test_metrics,
    }
    _write_metrics(output_dir / "test_metrics.json", payload)

    print(
        f"[{experiment_name} | {backbone} s{seed}] "
        f"test_acc={test_metrics['test_accuracy'] * 100:.2f}% "
        f"F1={test_metrics['test_f1_macro'] * 100:.2f}%"
    )
    del model, test_loader, ckpt
    _cleanup_cuda()
    return payload


def _fit_remaining_epochs(
    *,
    engine: TrainingEngine,
    start_epoch: int,
    epochs: int,
    output_dir: Path,
    name: str,
    early_stopping_patience: int,
    tracker: WandbTracker | None,
) -> dict[str, Any]:
    """Continue a TrainingEngine from a restored checkpoint.

    A last checkpoint is written after every completed epoch so a transient
    instance failure loses at most the currently-running epoch.
    """
    early_stopping_counter = 0
    early_stopped = False

    for epoch in range(start_epoch, epochs):
        engine.current_epoch = epoch
        train_metrics = engine.train_one_epoch(epoch, epochs)
        engine.train_history.append(train_metrics)

        val_metrics = engine.validate(epoch, epochs)
        engine.val_history.append(val_metrics)

        if engine.scheduler is not None:
            engine.scheduler.step()

        is_best = val_metrics["val_acc"] > engine.best_val_acc
        if is_best:
            engine.best_val_acc = val_metrics["val_acc"]
            engine.best_epoch = epoch
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if tracker is not None:
            tracker.log(
                {
                    "train/loss": train_metrics["train_loss"],
                    "train/acc": train_metrics["train_acc"],
                    "val/loss": val_metrics["val_loss"],
                    "val/acc": val_metrics["val_acc"],
                    "lr": engine.optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )

        lr = engine.optimizer.param_groups[0]["lr"]
        best_marker = "[Best]" if is_best else ""
        early_marker = (
            f"[EarlyStop {early_stopping_counter}/{early_stopping_patience}]"
        )
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_metrics['train_loss']:.4f}, "
            f"Acc: {train_metrics['train_acc']:.2f}% | "
            f"Val Loss: {val_metrics['val_loss']:.4f}, "
            f"Acc: {val_metrics['val_acc']:.2f}% | "
            f"LR: {lr:.2e}{best_marker}{early_marker}"
        )

        if is_best:
            engine.save_checkpoint(
                output_dir / f"{name}_best.pth",
                epoch=epoch,
                metrics={**train_metrics, **val_metrics},
            )
            print(
                f"  ↳ New best! Saved to {name}_best.pth "
                f"(Val Acc: {val_metrics['val_acc']:.2f}%)"
            )

        engine.save_checkpoint(
            output_dir / f"{name}_last.pth",
            epoch=epoch,
            metrics={**train_metrics, **val_metrics},
        )

        if early_stopping_counter >= early_stopping_patience:
            print(
                "\n[EarlyStopping] No improvement for "
                f"{early_stopping_patience} epochs. "
                f"Stopping at epoch {epoch + 1}."
            )
            early_stopped = True
            break

    return {
        "best_val_acc": engine.best_val_acc,
        "best_epoch": engine.best_epoch + 1,
        "final_train_acc": engine.train_history[-1]["train_acc"],
        "final_val_acc": engine.val_history[-1]["val_acc"],
        "stopped_early": early_stopped,
        "epochs_trained": engine.current_epoch + 1,
    }


def train_one(
    *,
    root: Path,
    backbone: str,
    seed: int,
    experiment_name: str,
    experiment_cfg: dict[str, Any],
    data_root: Path,
    batch_size: int,
    epochs: int = 50,
    device: str | None = None,
    resume_training: bool = False,
) -> dict[str, Any]:
    """Train and evaluate one Task B 17-class run."""
    root = Path(root)
    data_root = Path(data_root)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    output_dir = experiment_output_dir(root, experiment_name, seed, backbone)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "test_metrics.json"

    if metrics_path.exists():
        print(f"[{experiment_name} | {backbone} s{seed}] Reusing {metrics_path}")
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    resume_checkpoint: dict[str, Any] | None = None
    resume_checkpoint_path: Path | None = None
    if resume_training:
        checkpoint_name = f"{backbone}_seed{seed}"
        candidates = [
            output_dir / f"{checkpoint_name}_last.pth",
            output_dir / f"{checkpoint_name}_best.pth",
        ]
        resume_checkpoint_path = next(
            (path for path in candidates if path.exists()), None
        )
        if resume_checkpoint_path is not None:
            resume_checkpoint = _load_checkpoint(resume_checkpoint_path)
            start_epoch = int(resume_checkpoint.get("epoch", -1)) + 1
            if start_epoch >= epochs:
                resumed = _evaluate_existing_checkpoint(
                    backbone=backbone,
                    seed=seed,
                    experiment_name=experiment_name,
                    experiment_cfg=experiment_cfg,
                    data_root=data_root,
                    output_dir=output_dir,
                    batch_size=batch_size,
                    device=device,
                )
                if resumed is not None:
                    return resumed
    else:
        resumed = _evaluate_existing_checkpoint(
            backbone=backbone,
            seed=seed,
            experiment_name=experiment_name,
            experiment_cfg=experiment_cfg,
            data_root=data_root,
            output_dir=output_dir,
            batch_size=batch_size,
            device=device,
        )
        if resumed is not None:
            return resumed

    tracker = None
    tracker_finished = False
    try:
        train_loader, val_loader, test_loader = get_dataloaders(
            data_root=data_root,
            batch_size=batch_size,
            num_workers=4,
            task="taskB",
            use_90rotation=bool(experiment_cfg["use_90rotation"]),
        )
        class_names = list(train_loader.dataset.classes)
        num_classes = len(class_names)
        label_smoothing = float(experiment_cfg["label_smoothing"])

        print(
            f"[{experiment_name} | {backbone} s{seed}] "
            f"train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, "
            f"test={len(test_loader.dataset)}, classes={num_classes}"
        )

        run_config = {
            "task": "Task-B",
            "variant": "17class",
            "backbone": backbone,
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "num_classes": num_classes,
            "classes": class_names,
            "lr_head": 1e-4,
            "lr_backbone": 1e-5,
            "weight_decay": 1e-4,
            "label_smoothing": label_smoothing,
            "use_90rotation": bool(experiment_cfg["use_90rotation"]),
            "scheduler": "cosine",
            "amp": True,
            "pretrained": "ImageNet",
            "experiment": experiment_name,
        }
        _write_metrics(output_dir / "run_config.json", run_config)

        tracker = WandbTracker.start(
            project="woodbench-taskB-17class",
            name=f"{backbone}_{experiment_name}_s{seed}",
            config=run_config,
            tags=["final-B", backbone, f"seed{seed}", *experiment_cfg["tags"]],
            dir_path=str(output_dir),
        )

        model = create_model(
            backbone,
            num_classes=num_classes,
            pretrained=resume_checkpoint is None,
        )
        optimizer = create_optimizer(
            model,
            lr=1e-4,
            weight_decay=1e-4,
            backbone_lr=1e-5,
        )
        scheduler = create_scheduler(optimizer, epochs=epochs, scheduler_type="cosine")
        criterion = create_loss_function(
            "label_smoothing",
            label_smoothing=label_smoothing,
        )

        engine = TrainingEngine(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            use_amp=True,
        )
        if resume_checkpoint is not None:
            model.load_state_dict(resume_checkpoint["model_state_dict"])
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            if (
                scheduler is not None
                and "scheduler_state_dict" in resume_checkpoint
            ):
                scheduler.load_state_dict(
                    resume_checkpoint["scheduler_state_dict"]
                )
            engine.best_val_acc = float(
                resume_checkpoint.get("best_val_acc", 0.0)
            )
            engine.best_epoch = int(resume_checkpoint.get("best_epoch", 0))
            start_epoch = int(resume_checkpoint.get("epoch", -1)) + 1
            print(
                f"[Resume] {resume_checkpoint_path} -> "
                f"epoch {start_epoch + 1}/{epochs}"
            )
            result = _fit_remaining_epochs(
                engine=engine,
                start_epoch=start_epoch,
                epochs=epochs,
                output_dir=output_dir,
                name=f"{backbone}_seed{seed}",
                early_stopping_patience=10,
                tracker=tracker,
            )
        else:
            result = engine.fit(
                epochs=epochs,
                output_dir=output_dir,
                name=f"{backbone}_seed{seed}",
                save_best_only=True,
                early_stopping_patience=10,
                tracker=tracker,
            )

        test_metrics = evaluate_model(model, test_loader, device)
        payload = {
            "backbone": backbone,
            "seed": seed,
            "experiment": experiment_name,
            "num_classes": num_classes,
            "class_names": class_names,
            **result,
            **test_metrics,
        }

        if tracker:
            tracker.log(
                {
                    "best_epoch": result["best_epoch"],
                    "best_val_acc": result["best_val_acc"],
                    "epochs_trained": result["epochs_trained"],
                    "stopped_early": result["stopped_early"],
                    "test/accuracy": test_metrics["test_accuracy"],
                    "test/f1_macro": test_metrics["test_f1_macro"],
                    "test/precision_macro": test_metrics["test_precision_macro"],
                    "test/recall_macro": test_metrics["test_recall_macro"],
                    **{
                        f"test/f1/{class_name}": score
                        for class_name, score in test_metrics[
                            "test_f1_per_class"
                        ].items()
                    },
                },
                step=result["epochs_trained"],
            )
            tracker.finish()
            tracker_finished = True

        _write_metrics(metrics_path, payload)
        print(
            f"[{experiment_name} | {backbone} s{seed}] "
            f"val_acc={result['best_val_acc']:.2f}% "
            f"test_acc={test_metrics['test_accuracy'] * 100:.2f}% "
            f"F1={test_metrics['test_f1_macro'] * 100:.2f}%"
        )
        return payload
    finally:
        if tracker and not tracker_finished:
            tracker.finish()
        try:
            del engine
        except UnboundLocalError:
            pass
        try:
            del model
        except UnboundLocalError:
            pass
        try:
            del optimizer
        except UnboundLocalError:
            pass
        try:
            del scheduler
        except UnboundLocalError:
            pass
        try:
            del criterion
        except UnboundLocalError:
            pass
        try:
            del train_loader, val_loader, test_loader
        except UnboundLocalError:
            pass
        _cleanup_cuda()


def run_all_experiments(
    *,
    backbone_name: str,
    root: Path,
    data_root: Path | None = None,
    batch_size: int | None = None,
    epochs: int = 50,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """Run the 6 final Task B 17-class runs for a backbone."""
    if backbone_name not in BACKBONE_CONFIGS:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    root = Path(root)
    data_root = Path(data_root) if data_root else root / "data" / "task-B-group-split"
    batch_size = batch_size or int(BACKBONE_CONFIGS[backbone_name]["batch_size"])
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    verify_17class_dataset(data_root)
    print(f"[Device] {device}")
    print(f"[Output] {root / OUTPUT_EXPERIMENT_ROOT}")

    results: list[dict[str, Any]] = []
    failures: dict[str, str] = {}
    for experiment_name, experiment_cfg in EXPERIMENTS.items():
        for seed in experiment_cfg["seeds"]:
            run_id = f"{experiment_name}/seed_{seed}/{backbone_name}"
            print("\n" + "=" * 80)
            print(run_id)
            print("=" * 80)
            try:
                result = train_one(
                    root=root,
                    backbone=backbone_name,
                    seed=seed,
                    experiment_name=experiment_name,
                    experiment_cfg=experiment_cfg,
                    data_root=data_root,
                    batch_size=batch_size,
                    epochs=epochs,
                    device=device,
                )
                results.append(result)
            except Exception as exc:
                failures[run_id] = repr(exc)
                _cleanup_cuda()
                print(f"[ERROR] {run_id}: {exc!r}")

    if failures:
        raise RuntimeError(f"{backbone_name} failed runs: {failures}")
    return results


def summarize_results(results: list[dict[str, Any]]) -> None:
    """Print a compact summary table for notebook output."""
    if not results:
        print("No results.")
        return

    print("\n" + "=" * 88)
    print("Task B 17-class summary")
    print("=" * 88)
    print(
        f"{'experiment':<30s} {'seed':>6s} {'val_acc':>9s} "
        f"{'test_acc':>9s} {'f1':>9s} {'prec':>9s} {'rec':>9s}"
    )
    print("-" * 88)
    for row in results:
        print(
            f"{row['experiment']:<30s} {row['seed']:>6d} "
            f"{row['best_val_acc']:>8.2f}% "
            f"{row['test_accuracy'] * 100:>8.2f}% "
            f"{row['test_f1_macro'] * 100:>8.2f}% "
            f"{row['test_precision_macro'] * 100:>8.2f}% "
            f"{row['test_recall_macro'] * 100:>8.2f}%"
        )

    main = [row for row in results if row["experiment"] == "main"]
    if len(main) > 1:
        accs = [row["test_accuracy"] * 100 for row in main]
        f1s = [row["test_f1_macro"] * 100 for row in main]
        print("-" * 88)
        print(f"main test_acc: {np.mean(accs):.2f} +/- {np.std(accs):.2f}%")
        print(f"main test_f1:  {np.mean(f1s):.2f} +/- {np.std(f1s):.2f}%")
