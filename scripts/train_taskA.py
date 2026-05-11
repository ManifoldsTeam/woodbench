#!/usr/bin/env python3
"""Training script for Task A: Conifer Family Classification.

Usage:
    python scripts/train_taskA.py --backbone resnet50 --epochs 50
    python scripts/train_taskA.py --backbone efficientnet_b3 --epochs 50 --batch_size 32
    python scripts/train_taskA.py --all_models --epochs 50
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import get_dataloaders
from src.models import create_model, BACKBONE_NAMES
from src.train import TrainingEngine, create_optimizer, create_scheduler, create_criterion
from src.utils import set_seed, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Conifer Family Classification (Task A)"
    )
    
    # Model
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        choices=list(BACKBONE_NAMES.keys()),
        help="Backbone architecture",
    )
    parser.add_argument(
        "--all_models",
        action="store_true",
        help="Train all 5 models sequentially",
    )
    
    # Data
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/split",
        help="Path to dataset root",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Image size",
    )
    
    # Training
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step"])
    
    # Augmentation
    parser.add_argument(
        "--use_90rotation",
        action="store_true",
        help="Use 90-degree rotation augmentation",
    )

    # Fine-tuning (unfreeze backbone)
    parser.add_argument(
        "--finetune",
        action="store_true",
        help=(
            "Unfreeze backbone and fine-tune all layers. "
            "Uses differential learning rates: backbone_lr=1e-5, head_lr=1e-4."
        ),
    )
    parser.add_argument(
        "--backbone_lr",
        type=float,
        default=1e-5,
        help=(
            "Learning rate for backbone when finetune=True. "
            "Head LR is controlled by --lr. Default: 1e-5 (1/10 of head LR)."
        ),
    )

    # Hardware
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/taskA",
        help="Output directory",
    )
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    
    return parser.parse_args()


def train_single_model(
    backbone: str,
    args,
    output_dir: Path,
    logger,
) -> dict:
    """Train a single model.
    
    Returns:
        Dictionary with training results
    """
    # Set seed
    set_seed(args.seed)
    
    # Create experiment directory
    exp_dir = output_dir / backbone / f"seed_{args.seed}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"=" * 60)
    logger.info(f"Training {backbone} with seed={args.seed}")
    logger.info(f"=" * 60)
    
    # Create dataloaders
    logger.info("Loading dataset...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_90rotation=args.use_90rotation,
        img_size=args.img_size,
    )
    
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    logger.info(f"Creating model: {backbone}")
    freeze_backbone = not args.finetune
    logger.info(f"  Mode: {'fine-tuning (backbone unfrozen)' if args.finetune else 'linear probing (backbone frozen)'}")
    model = create_model(backbone, num_classes=4, pretrained=True, freeze_backbone=freeze_backbone)

    # Create optimizer
    backbone_lr = args.backbone_lr if args.finetune else None
    if args.finetune:
        logger.info(f"  Head LR: {args.lr}, Backbone LR: {backbone_lr}")
    optimizer = create_optimizer(model, lr=args.lr, weight_decay=args.weight_decay, backbone_lr=backbone_lr)
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, epochs=args.epochs, scheduler_type=args.scheduler)
    
    # Create criterion
    criterion = create_criterion(label_smoothing=args.label_smoothing)
    
    # Create training engine
    engine = TrainingEngine(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=args.device,
        use_amp=not args.no_amp,
    )
    
    # Train
    logger.info(f"Starting training for {args.epochs} epochs...")
    results = engine.fit(
        epochs=args.epochs,
        output_dir=exp_dir,
        name=f"{backbone}",
        save_best_only=False,
        verbose=True,
    )
    
    # Save experiment config
    config = {
        "backbone": backbone,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "scheduler": args.scheduler,
        "use_90rotation": args.use_90rotation,
        "seed": args.seed,
        "img_size": args.img_size,
        "num_workers": args.num_workers,
        "use_amp": not args.no_amp,
        # Fine-tuning
        "finetune": args.finetune,
        "freeze_backbone": not args.finetune,
        "backbone_lr": args.backbone_lr if args.finetune else None,
        "results": results,
    }
    
    config_path = exp_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Training complete!")
    logger.info(f"Best val accuracy: {results['best_val_acc']:.2f}%")
    logger.info(f"Checkpoints saved to: {exp_dir}")
    
    return results


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    
    # Setup logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = args.name or f"taskA_{timestamp}"
    logger = setup_logger(name, output_dir / "logs")
    
    logger.info("Task A: Conifer Family Classification")
    logger.info(f"Arguments: {vars(args)}")
    
    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = "cpu"
    
    if args.device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    
    results = {}
    
    if args.all_models:
        # Train all models
        for backbone in BACKBONE_NAMES.keys():
            try:
                results[backbone] = train_single_model(backbone, args, output_dir, logger)
            except Exception as e:
                logger.error(f"Error training {backbone}: {e}")
                results[backbone] = {"error": str(e)}
    else:
        # Train single model
        results[args.backbone] = train_single_model(args.backbone, args, output_dir, logger)
    
    # Save final results
    results_path = output_dir / f"{name}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {results_path}")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    for backbone, result in results.items():
        if "error" in result:
            logger.info(f"{backbone}: ERROR - {result['error']}")
        else:
            logger.info(f"{backbone}: Best Val Acc = {result.get('best_val_acc', 0):.2f}%")


if __name__ == "__main__":
    main()
