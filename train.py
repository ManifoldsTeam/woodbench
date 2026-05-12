"""Training entry point for WoodID-S2-2026.

Supports all experiments via Hydra config composition:
    python train.py experiment=taskA model.backbone=resnet50
    python train.py experiment=taskA model.backbone=efficientnet_b3
    python train.py --multirun model.backbone=[resnet50,efficientnet_b3,mobilenetv3_large]

For multi-seed runs:
    python train.py train.num_seeds=3
"""

import os
import sys
from pathlib import Path

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
from src.train import CheckpointCallback
from src.utils import MetricsLogger


def build_transforms(cfg: DictConfig):
    """Build train/val transforms from config."""
    aug_cfg = cfg.get("augmentation", {})

    use_90rotation = aug_cfg.get("rotation", {}).get("type") == "discrete_90"
    degrees = aug_cfg.get("rotation", {}).get("degrees", 30)
    hflip = aug_cfg.get("horizontal_flip", 0.5)
    vflip = aug_cfg.get("vertical_flip", 0.5)
    brightness = aug_cfg.get("color_jitter", {}).get("brightness", 0.2)
    contrast = aug_cfg.get("color_jitter", {}).get("contrast", 0.2)

    img_size = cfg.get("image", {}).get("size", 224)

    train_transform = get_train_transforms(
        img_size=img_size,
        use_90rotation=use_90rotation,
        degrees=degrees,
        hflip_prob=hflip,
        vflip_prob=vflip,
        brightness=brightness,
        contrast=contrast,
    )
    val_transform = get_val_transforms(img_size=img_size)

    return train_transform, val_transform


def build_dataloaders(cfg: DictConfig, train_transform, val_transform):
    """Build train/val/test dataloaders."""
    from torch.utils.data import DataLoader

    data_root = Path(cfg.get("data_root", "data/"))
    batch_size = cfg.get("dataloader", {}).get("batch_size", 64)
    num_workers = cfg.get("dataloader", {}).get("num_workers", 4)
    pin_memory = cfg.get("dataloader", {}).get("pin_memory", True)

    train_dataset = ConiferDataset(
        data_root=data_root,
        split="train",
        transform=train_transform,
        task=cfg.get("task", {}).get("id", "taskA"),
    )
    val_dataset = ConiferDataset(
        data_root=data_root,
        split="val",
        transform=val_transform,
        task=cfg.get("task", {}).get("id", "taskA"),
    )
    test_dataset = ConiferDataset(
        data_root=data_root,
        split="test",
        transform=val_transform,
        task=cfg.get("task", {}).get("id", "taskA"),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2,
    )

    return train_loader, val_loader, test_loader


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main training loop."""
    print(OmegaConf.to_yaml(cfg))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_seeds = cfg.get("train", {}).get("num_seeds", 1)
    seeds = [cfg.get("train", {}).get("seed", 42) + i * 1 for i in range(num_seeds)]

    model_cfg = cfg.get("model", {})
    finetune_cfg = model_cfg.get("finetune", {})
    train_cfg = cfg.get("train", {})
    optimizer_cfg = train_cfg.get("optimizer", {})
    scheduler_cfg = train_cfg.get("scheduler", {})

    train_transform, val_transform = build_transforms(cfg)
    train_loader, val_loader, test_loader = build_dataloaders(cfg, train_transform, val_transform)

    results_all_seeds = []

    for seed_idx, seed in enumerate(seeds):
        set_seed(seed)

        model = create_model(
            backbone_name=model_cfg.get("backbone", "resnet50"),
            num_classes=model_cfg.get("num_classes", 4),
            pretrained=model_cfg.get("pretrained", True),
            freeze_backbone=finetune_cfg.get("freeze_backbone", False),
        )

        backbone_lr = finetune_cfg.get("backbone_lr", None)
        if backbone_lr is None:
            backbone_lr = optimizer_cfg.get("lr", 1e-4)

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

        label_smoothing = model_cfg.get("label_smoothing", 0.1)
        criterion = create_loss_function(
            loss_type="label_smoothing",
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
            use_amp=train_cfg.get("amp", True),
        )

        output_dir = Path(f"logs/{cfg.get('experiment_name', 'experiment')}/{model_cfg.get('backbone')}/seed_{seed}")
        output_dir.mkdir(parents=True, exist_ok=True)

        logger = setup_logger(log_dir=output_dir)
        metrics_logger = MetricsLogger(output_dir=output_dir)

        logger.info(f"Training seed {seed_idx + 1}/{num_seeds} (seed={seed})")
        logger.info(f"Model: {model_cfg.get('backbone')}, Task: {cfg.get('task', {}).get('id')}")
        logger.info(f"Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples")

        checkpoint_cb = CheckpointCallback(
            output_dir=output_dir,
            name=f"{model_cfg.get('backbone')}_seed{seed}",
            save_best_only=train_cfg.get("checkpoint", {}).get("save_best_only", True),
            metric=train_cfg.get("checkpoint", {}).get("metric", "val_acc"),
            mode=train_cfg.get("checkpoint", {}).get("mode", "max"),
        )

        for epoch in range(train_cfg.get("epochs", 50)):
            train_metrics = engine.train_one_epoch()
            val_metrics = engine.validate()

            if scheduler is not None:
                scheduler.step()

            if checkpoint_cb(
                model=engine.model,
                optimizer=engine.optimizer,
                epoch=epoch,
                metrics={**train_metrics, **val_metrics},
                scheduler=scheduler,
            ):
                logger.info(f"  Saved best model (val_acc={val_metrics['val_acc']:.2f}%)")

            metrics_logger.log(
                epoch=epoch + 1,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                lr=optimizer.param_groups[0]["lr"],
            )

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{train_cfg.get('epochs', 50)} | "
                    f"Train: loss={train_metrics['train_loss']:.4f}, acc={train_metrics['train_acc']:.2f}% | "
                    f"Val: loss={val_metrics['val_loss']:.4f}, acc={val_metrics['val_acc']:.2f}%"
                )

        logger.info(f"Seed {seed} complete. Best val_acc: {engine.best_val_acc:.2f}%")
        results_all_seeds.append({"seed": seed, "best_val_acc": engine.best_val_acc})

    if len(results_all_seeds) > 1:
        import numpy as np
        accs = [r["best_val_acc"] for r in results_all_seeds]
        logger.info(
            f"Multi-seed summary: {np.mean(accs):.2f} +/- {np.std(accs):.2f}% "
            f"(seeds: {seeds})"
        )

    print(f"\nTraining complete. Results: {results_all_seeds}")


if __name__ == "__main__":
    main()
