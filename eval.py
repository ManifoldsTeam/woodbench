"""Evaluation entry point for WoodID-S2-2026.

Evaluate trained models with comprehensive metrics:
    python eval.py experiment=taskA model.backbone=resnet50 checkpoint=logs/.../best.pth
    python eval.py experiment=taskA checkpoint=logs/best_model.pth
"""

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src import (
    ConiferDataset,
    create_model,
    get_val_transforms,
    compute_metrics,
    plot_confusion_matrix,
    setup_logger,
)
from src.eval.metrics import measure_inference_time


def build_dataloader(cfg: DictConfig, split: str = "test"):
    """Build a single split dataloader."""
    from torch.utils.data import DataLoader

    data_root = Path(cfg.get("data_root", "data/"))
    val_transform = get_val_transforms(img_size=cfg.get("image", {}).get("size", 224))

    dataset = ConiferDataset(
        data_root=data_root,
        split=split,
        transform=val_transform,
        task=cfg.get("task", {}).get("id", "taskA"),
    )

    return DataLoader(
        dataset,
        batch_size=cfg.get("dataloader", {}).get("batch_size", 64),
        shuffle=False,
        num_workers=cfg.get("dataloader", {}).get("num_workers", 4),
        pin_memory=True,
    )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main evaluation loop."""
    print(OmegaConf.to_yaml(cfg))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = setup_logger()

    model_cfg = cfg.get("model", {})
    task_cfg = cfg.get("task", {})
    num_classes = model_cfg.get("num_classes", 4)

    family_classes = model_cfg.get("family_classes", None)
    if family_classes is None:
        family_classes = [f"class_{i}" for i in range(num_classes)]

    model = create_model(
        backbone_name=model_cfg.get("backbone", "resnet50"),
        num_classes=num_classes,
        pretrained=False,
    )

    checkpoint_path = cfg.get("checkpoint", None)
    if checkpoint_path is None:
        logger.warning("No checkpoint provided. Evaluating random initialization.")
    else:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded checkpoint: {checkpoint_path}")
            if "metrics" in checkpoint:
                logger.info(f"Checkpoint val_acc: {checkpoint['metrics'].get('val_acc', 'N/A')}")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")

    model = model.to(device)
    model.eval()

    logger.info("Evaluating on test set...")
    test_loader = build_dataloader(cfg, split="test")
    val_loader = build_dataloader(cfg, split="val")

    logger.info(f"Test samples: {len(test_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")

    test_metrics = compute_metrics(
        model=model,
        data_loader=test_loader,
        device=device,
        num_classes=num_classes,
        class_names=family_classes,
    )

    val_metrics = compute_metrics(
        model=model,
        data_loader=val_loader,
        device=device,
        num_classes=num_classes,
        class_names=family_classes,
    )

    logger.info("=" * 50)
    logger.info(f"Model: {model_cfg.get('backbone')}, Task: {task_cfg.get('id')}")
    logger.info("=" * 50)
    logger.info("TEST SET:")
    logger.info(f"  Accuracy:      {test_metrics['accuracy'] * 100:.2f}%")
    logger.info(f"  Precision:     {test_metrics['precision_macro'] * 100:.2f}%")
    logger.info(f"  Recall:        {test_metrics['recall_macro'] * 100:.2f}%")
    logger.info(f"  F1-macro:      {test_metrics['f1_macro'] * 100:.2f}%")
    logger.info("")
    logger.info("VAL SET:")
    logger.info(f"  Accuracy:      {val_metrics['accuracy'] * 100:.2f}%")
    logger.info(f"  F1-macro:      {val_metrics['f1_macro'] * 100:.2f}%")

    if "per_class" in test_metrics:
        logger.info("")
        logger.info("Per-class F1 (Test):")
        for name, scores in test_metrics["per_class"].items():
            logger.info(f"  {name:20s}: {scores['f1'] * 100:.2f}%")

    if cfg.get("evaluation", {}).get("confusion_matrix", False):
        output_dir = Path(cfg.get("checkpoint", "eval_output")).parent
        cm_path = output_dir / f"{model_cfg.get('backbone')}_confusion_matrix.png"
        plot_confusion_matrix(
            cm=test_metrics["confusion_matrix"],
            class_names=family_classes,
            normalize=True,
            title=f"{model_cfg.get('backbone')} — Confusion Matrix",
        )
        from src.eval.confusion import save_confusion_matrix
        save_confusion_matrix(
            cm=test_metrics["confusion_matrix"],
            class_names=family_classes,
            output_path=cm_path,
            normalize=True,
            title=f"{model_cfg.get('backbone')} — Test Set",
        )
        logger.info(f"Confusion matrix saved: {cm_path}")

    if device == "cuda":
        logger.info("")
        logger.info("Measuring inference time...")
        timing = measure_inference_time(
            model=model,
            device=device,
            num_runs=100,
            batch_size=1,
        )
        logger.info(f"  Mean: {timing['mean_ms']:.2f} ms/image")
        logger.info(f"  Std:  {timing['std_ms']:.2f} ms")
        logger.info(f"  Min:  {timing['min_ms']:.2f} ms")
        logger.info(f"  Max:  {timing['max_ms']:.2f} ms")


if __name__ == "__main__":
    main()
