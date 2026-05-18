#!/usr/bin/env python3
"""Evaluation script for Task A: Conifer Family Classification.

Usage:
    python scripts/eval_taskA.py --checkpoint experiments/taskA/resnet50/seed_42/resnet50_best.pth
    python scripts/eval_taskA.py --checkpoint experiments/taskA/resnet50/seed_42/resnet50_best.pth --save_cm
"""

import argparse
import json
from pathlib import Path

import torch

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import get_dataloaders
from src.models import create_model
from src.eval import compute_metrics, format_metrics, save_confusion_matrix, plot_per_class_f1
from src.eval.metrics import measure_inference_time
from src.utils import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Conifer Family Classification (Task A)"
    )
    
    # Model
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        help="Backbone architecture (for Grad-CAM)",
    )
    
    # Data
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/split",
        help="Path to dataset root",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    
    # Hardware
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    
    # Output
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--save_cm", action="store_true", help="Save confusion matrix")
    parser.add_argument("--save_f1", action="store_true", help="Save F1 bar chart")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup paths
    checkpoint_path = Path(args.checkpoint)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = checkpoint_path.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = setup_logger("eval", output_dir)
    
    logger.info("Task A: Conifer Family Classification - Evaluation")
    
    # Class names
    CLASS_NAMES = ["Cupressaceae", "Pinaceae", "Podocarpaceae", "Taxaceae"]
    
    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = "cpu"
    
    # Load model
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    model = create_model(args.backbone, num_classes=4, pretrained=False)
    
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(args.device)
    model.eval()
    
    # Load data
    logger.info("Loading dataset...")
    _, val_loader, test_loader = get_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    metrics = compute_metrics(
        model=model,
        data_loader=test_loader,
        device=args.device,
        num_classes=4,
        class_names=CLASS_NAMES,
    )
    
    # Print results
    print(format_metrics(metrics, CLASS_NAMES))
    
    # Save results
    results_path = output_dir / "test_results.json"
    with open(results_path, "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        json_metrics = {k: v.tolist() if hasattr(v, "tolist") else v for k, v in metrics.items()}
        json.dump(json_metrics, f, indent=2)
    
    logger.info(f"Results saved to: {results_path}")
    
    # Save confusion matrix
    if args.save_cm:
        cm_path = output_dir / "confusion_matrix.png"
        save_confusion_matrix(
            cm=metrics["confusion_matrix"],
            class_names=CLASS_NAMES,
            output_path=cm_path,
            title="Confusion Matrix - Task A (Test Set)",
        )
        logger.info(f"Confusion matrix saved to: {cm_path}")
    
    # Save per-class F1
    if args.save_f1:
        f1_path = output_dir / "per_class_f1.png"
        plot_per_class_f1(
            per_class_metrics=metrics["per_class"],
            class_names=CLASS_NAMES,
            output_path=f1_path,
        )
        logger.info(f"Per-class F1 saved to: {f1_path}")
    
    # Measure inference time (if GPU)
    if args.device == "cuda":
        logger.info("Measuring inference time...")
        timing = measure_inference_time(model, device=args.device, num_runs=100)
        logger.info(f"Inference time: {timing['mean_ms']:.2f} ms/batch, {timing['per_image_ms']:.2f} ms/image")
        
        timing_path = output_dir / "inference_time.json"
        with open(timing_path, "w") as f:
            json.dump(timing, f, indent=2)
    
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
