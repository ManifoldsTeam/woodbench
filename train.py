"""Training entry point (Hydra). Config: configs/ — no hyperparameters hardcoded here.

Examples:
    python train.py experiment=taskB model.backbone=resnet50 train.seed=42
    python train.py experiment=taskB_90rotation model.backbone=convnext_tiny
    python train.py experiment=taskA model.backbone=resnet50 train.wandb.enabled=false
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src import (
    ConiferDataset,
    TrainingEngine,
    create_loss_function,
    create_model,
    create_optimizer,
    create_scheduler,
    get_train_transforms,
    get_val_transforms,
    set_seed,
    setup_logger,
)
from src.train import CheckpointCallback, WandbTracker
from src.utils import MetricsLogger


def _image_size(cfg: DictConfig) -> int:
    return int(
        OmegaConf.select(cfg, "image.size")
        or OmegaConf.select(cfg, "data.image.size")
        or 224
    )


def _augmentation_cfg(cfg: DictConfig) -> Dict[str, Any]:
    aug = OmegaConf.select(cfg, "augmentation") or OmegaConf.select(
        cfg, "train.augmentation"
    )
    return OmegaConf.to_container(aug, resolve=True) if aug else {}


def build_transforms(cfg: DictConfig) -> Tuple[Any, Any]:
    """Build train/val transforms from Hydra config."""
    aug_cfg = _augmentation_cfg(cfg)
    rotation = aug_cfg.get("rotation", {})
    color = aug_cfg.get("color_jitter", {})
    img_size = _image_size(cfg)

    train_transform = get_train_transforms(
        img_size=img_size,
        use_90rotation=rotation.get("type") == "discrete_90",
        degrees=rotation.get("degrees", 30),
        hflip_prob=aug_cfg.get("horizontal_flip", 0.5),
        vflip_prob=aug_cfg.get("vertical_flip", 0.5),
        brightness=color.get("brightness", 0.2),
        contrast=color.get("contrast", 0.2),
    )
    val_transform = get_val_transforms(img_size=img_size)
    return train_transform, val_transform


def build_dataloaders(cfg: DictConfig, train_transform, val_transform):
    """Build train/val/test dataloaders."""
    from torch.utils.data import DataLoader

    data_root = Path(cfg.get("data_root", "data"))
    loader_cfg = OmegaConf.to_container(
        OmegaConf.select(cfg, "data.dataloader") or {},
        resolve=True,
    )
    batch_size = loader_cfg.get("batch_size", 64)
    num_workers = loader_cfg.get("num_workers", 4)
    pin_memory = loader_cfg.get("pin_memory", True)
    task_id = cfg.get("task", {}).get("id", "taskA")

    train_dataset = ConiferDataset(
        data_root=data_root,
        split="train",
        transform=train_transform,
        task=task_id,
    )
    val_dataset = ConiferDataset(
        data_root=data_root,
        split="val",
        transform=val_transform,
        task=task_id,
    )
    test_dataset = ConiferDataset(
        data_root=data_root,
        split="test",
        transform=val_transform,
        task=task_id,
    )

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
        "prefetch_factor": 2 if num_workers > 0 else None,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        **common,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, shuffle=False, **common)

    return train_loader, val_loader, test_loader


def experiment_output_dir(cfg: DictConfig, seed: int) -> Path:
    """Stable layout for checkpoints (rclone-friendly)."""
    task_id = cfg.get("task", {}).get("id", "task")
    backbone = cfg.get("model", {}).get("backbone", "model")
    root = Path(cfg.get("output", {}).get("root", "experiments"))
    return root / task_id / backbone / f"seed_{seed}"


def _start_wandb(cfg: DictConfig, seed: int, output_dir: Path) -> WandbTracker | None:
    wb = cfg.get("train", {}).get("wandb", {})
    if not wb.get("enabled", False):
        return None

    tags = list(wb.get("tags", []))
    tags.append(cfg.get("task", {}).get("id", "task"))

    return WandbTracker.start(
        project=wb.get("project", "woodbench"),
        entity=wb.get("entity"),
        name=f"{cfg.get('experiment_name', 'run')}_{cfg.model.backbone}_s{seed}",
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=tags,
        dir_path=str(output_dir),
    )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training loop."""
    print(OmegaConf.to_yaml(cfg))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_cfg = cfg.get("train", {})
    num_seeds = int(train_cfg.get("num_seeds", 1))
    base_seed = int(train_cfg.get("seed", 42))
    seeds = [base_seed + i for i in range(num_seeds)]

    model_cfg = cfg.get("model", {})
    finetune_cfg = model_cfg.get("finetune", {})
    optimizer_cfg = train_cfg.get("optimizer", {})
    scheduler_cfg = train_cfg.get("scheduler", {})
    checkpoint_cfg = train_cfg.get("checkpoint", {})
    early_cfg = train_cfg.get("early_stopping", {})

    train_transform, val_transform = build_transforms(cfg)
    train_loader, val_loader, _test_loader = build_dataloaders(
        cfg, train_transform, val_transform
    )

    results_all_seeds = []

    for seed_idx, seed in enumerate(seeds):
        set_seed(seed)
        output_dir = experiment_output_dir(cfg, seed)
        output_dir.mkdir(parents=True, exist_ok=True)

        OmegaConf.save(cfg, output_dir / "config.yaml")

        logger = setup_logger("woodbench", log_dir=output_dir)
        metrics_logger = MetricsLogger(output_dir=output_dir)
        tracker = _start_wandb(cfg, seed, output_dir)

        model = create_model(
            backbone_name=model_cfg.get("backbone", "resnet50"),
            num_classes=model_cfg.get("num_classes", 4),
            pretrained=model_cfg.get("pretrained", True),
            freeze_backbone=finetune_cfg.get("freeze_backbone", False),
        )

        backbone_lr = finetune_cfg.get("backbone_lr")
        optimizer = create_optimizer(
            model=model,
            lr=optimizer_cfg.get("lr", 1e-4),
            weight_decay=optimizer_cfg.get("weight_decay", 1e-4),
            backbone_lr=backbone_lr,
        )

        scheduler = create_scheduler(
            optimizer=optimizer,
            epochs=train_cfg.get("epochs", 50),
            scheduler_type=scheduler_cfg.get("type", "cosine"),
        )

        criterion = create_loss_function(
            loss_type="label_smoothing",
            label_smoothing=model_cfg.get("label_smoothing", 0.1),
        )

        engine = TrainingEngine(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            use_amp=train_cfg.get("amp", True),
        )

        backbone = model_cfg.get("backbone", "model")
        logger.info(f"Seed {seed_idx + 1}/{num_seeds} (seed={seed})")
        logger.info(f"Output: {output_dir}")
        logger.info(
            f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}"
        )

        checkpoint_cb = CheckpointCallback(
            output_dir=output_dir,
            name=f"{backbone}_seed{seed}",
            save_best_only=checkpoint_cfg.get("save_best_only", True),
            metric=checkpoint_cfg.get("metric", "val_acc"),
            mode=checkpoint_cfg.get("mode", "max"),
        )

        epochs = int(train_cfg.get("epochs", 50))
        patience = (
            int(early_cfg.get("patience", 10))
            if early_cfg.get("enabled", False)
            else 0
        )

        for epoch in range(epochs):
            train_metrics = engine.train_one_epoch(epoch, epochs)
            val_metrics = engine.validate(epoch, epochs)

            if scheduler is not None:
                scheduler.step()

            if val_metrics["val_acc"] > engine.best_val_acc:
                engine.best_val_acc = val_metrics["val_acc"]
                engine.best_epoch = epoch

            combined = {**train_metrics, **val_metrics}
            checkpoint_cb(
                model=engine.model,
                optimizer=engine.optimizer,
                epoch=epoch,
                metrics=combined,
                scheduler=scheduler,
            )

            metrics_logger.log(
                epoch=epoch + 1,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                lr=optimizer.param_groups[0]["lr"],
            )

            if tracker is not None:
                tracker.log(
                    {
                        "train/loss": train_metrics["train_loss"],
                        "train/acc": train_metrics["train_acc"],
                        "val/loss": val_metrics["val_loss"],
                        "val/acc": val_metrics["val_acc"],
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    step=epoch,
                )

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"train loss={train_metrics['train_loss']:.4f} "
                    f"acc={train_metrics['train_acc']:.2f}% | "
                    f"val loss={val_metrics['val_loss']:.4f} "
                    f"acc={val_metrics['val_acc']:.2f}%"
                )

        if tracker is not None:
            tracker.finish()

        logger.info(f"Seed {seed} done. Best val_acc: {engine.best_val_acc:.2f}%")
        results_all_seeds.append({"seed": seed, "best_val_acc": engine.best_val_acc})

    if len(results_all_seeds) > 1:
        import numpy as np

        accs = [r["best_val_acc"] for r in results_all_seeds]
        logger.info(
            f"Multi-seed: {np.mean(accs):.2f} +/- {np.std(accs):.2f}% (seeds={seeds})"
        )

    print(f"Training complete. Results: {results_all_seeds}")


if __name__ == "__main__":
    main()
