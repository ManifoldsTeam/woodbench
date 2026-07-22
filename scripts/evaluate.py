#!/usr/bin/env python3
"""Evaluation script for Task B: Species Classification.

This script evaluates trained models and generates visualizations.

Usage:
    python scripts/evaluate.py --checkpoint experiments/taskB/convnext_tiny/seed_42/best.pth
    python scripts/evaluate.py --experiment_dir experiments/taskB/convnext_tiny/
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import get_dataloaders
from src.eval.metrics import compute_metrics
from src.models import create_model
from src.utils import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Task B models"
    )

    # Model path
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        help="Path to experiment directory (will use best.pth)",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="convnext_tiny",
        help="Backbone name (needed if no checkpoint provided)",
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

    # Hardware
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: checkpoint parent)",
    )

    return parser.parse_args()


@torch.no_grad()
def evaluate_model(model, data_loader, device, num_classes):
    """Evaluate model and return predictions."""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in data_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs),
    )


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    output_path: Path,
    normalize: bool = False,
):
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(figsize=(16, 14))

    if normalize:
        title = "Normalized Confusion Matrix"
    else:
        title = "Confusion Matrix"

    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=class_names,
        normalize="true" if normalize else None,
        cmap="Blues",
        ax=ax,
        xticks_rotation=45,
        values_format=".2f" if normalize else "d",
    )

    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    plt.tight_layout()

    # Save both versions
    fig.savefig(output_path / "confusion_matrix.png", dpi=150, bbox_inches="tight")

    if normalize:
        fig.savefig(output_path / "confusion_matrix_normalized.png", dpi=150, bbox_inches="tight")

    plt.close(fig)


def plot_per_class_f1(
    metrics: dict,
    class_names: list,
    output_path: Path,
):
    """Plot per-class F1 scores as bar chart."""
    per_class_f1 = metrics["per_class"]

    # Group by family
    family_colors = {
        "Cupressaceae": "#1f77b4",
        "Pinaceae": "#ff7f0e",
        "Podocarpaceae": "#2ca02c",
        "Taxaceae": "#d62728",
    }

    fig, ax = plt.subplots(figsize=(16, 8))

    # Define species to family mapping (from WoodID_plan.md)
    species_to_family = {
        # Cupressaceae
        "Cunninghamia_lanceolata": "Cupressaceae",
        "Callitris_columellaris": "Cupressaceae",
        "Fokienia_hodginsii": "Cupressaceae",
        "Cupressus_funebris": "Cupressaceae",
        "Calocedrus_sp.": "Cupressaceae",
        "Cryptomeria_japonica": "Cupressaceae",
        "Xanthocyparis_vietnamensis": "Cupressaceae",
        "Chamaecyparis_obtusa": "Cupressaceae",
        "Calocedrus_rupestris": "Cupressaceae",
        "Thujopsis_dolabrata": "Cupressaceae",
        # Pinaceae
        "Pinus_dalatensis": "Pinaceae",
        "Pinus_radiata": "Pinaceae",
        "Pinus_sp.": "Pinaceae",
        "Pinus_krempfii": "Pinaceae",
        "Picea_luca": "Pinaceae",
        "Araucaria_araucana": "Pinaceae",
        "Pseudotsuga_brevifolia": "Pinaceae",
        "Pinus_cernua": "Pinaceae",
        # Podocarpaceae
        "Podocarpus_pilgeri": "Podocarpaceae",
        # Taxaceae
        "Taxus_wallichiana": "Taxaceae",
    }

    f1_scores = [per_class_f1[cls]["f1"] for cls in class_names]
    colors = [
        family_colors.get(species_to_family.get(cls, "Unknown"), "#808080")
        for cls in class_names
    ]

    bars = ax.bar(range(len(f1_scores)), f1_scores, color=colors, edgecolor="black")

    # Add value labels on bars
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        ax.annotate(
            f"{score:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Per-Class F1 Scores (Task B: Species Classification)", fontsize=14)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=np.mean(f1_scores), color="red", linestyle="--", label=f"Mean F1: {np.mean(f1_scores):.3f}")
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_path / "per_class_f1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curves(
    train_history: list,
    val_history: list,
    output_path: Path,
):
    """Plot training and validation learning curves."""
    train_loss = [h["train_loss"] for h in train_history]
    train_acc = [h["train_acc"] for h in train_history]
    val_loss = [h["val_loss"] for h in val_history]
    val_acc = [h["val_acc"] for h in val_history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(train_loss, label="Train Loss", color="blue")
    axes[0].plot(val_loss, label="Val Loss", color="orange")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(train_acc, label="Train Acc", color="blue")
    axes[1].plot(val_acc, label="Val Acc", color="orange")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy Curves")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path / "learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    elif args.experiment_dir:
        exp_dir = Path(args.experiment_dir)
        checkpoint_path = exp_dir / "best.pth"
    else:
        print("Error: Must provide --checkpoint or --experiment_dir")
        sys.exit(1)

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = checkpoint_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = setup_logger("evaluate", output_dir)

    logger.info(f"Evaluating checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})

    # Get config values
    backbone = config.get("backbone", args.backbone)
    num_classes = config.get("num_classes", 20)

    logger.info(f"Backbone: {backbone}")
    logger.info(f"Num classes: {num_classes}")

    # Create model
    model = create_model(backbone, num_classes=num_classes, pretrained=False, freeze_backbone=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"
    model = model.to(device)

    # Load data
    logger.info("Loading dataset...")
    _, val_loader, test_loader = get_dataloaders(
        data_root=args.data_root,
        batch_size=64,
        num_workers=4,
        img_size=args.img_size,
    )

    class_names = test_loader.dataset.classes
    logger.info(f"Classes: {class_names}")

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    preds, labels, probs = evaluate_model(model, test_loader, device, num_classes)

    # Compute metrics
    metrics = compute_metrics(model, test_loader, device, num_classes)

    logger.info(f"\nTest Accuracy: {metrics['accuracy'] * 100:.2f}%")
    logger.info(f"Test Precision (macro): {metrics['precision_macro'] * 100:.2f}%")
    logger.info(f"Test Recall (macro): {metrics['recall_macro'] * 100:.2f}%")
    logger.info(f"Test F1 (macro): {metrics['f1_macro'] * 100:.2f}%")

    # Classification report
    report = classification_report(labels, preds, target_names=class_names, zero_division=0)
    logger.info(f"\nClassification Report:\n{report}")

    # Save metrics
    metrics_path = output_dir / "test_metrics.json"
    metrics_to_save = {
        "accuracy": float(metrics["accuracy"]),
        "precision_macro": float(metrics["precision_macro"]),
        "recall_macro": float(metrics["recall_macro"]),
        "f1_macro": float(metrics["f1_macro"]),
        "per_class_f1": {
            cls: float(data["f1"])
            for cls, data in metrics["per_class"].items()
        },
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_to_save, f, indent=2)

    logger.info(f"\nMetrics saved to: {metrics_path}")

    # Generate visualizations
    logger.info("Generating visualizations...")

    # Confusion matrix
    plot_confusion_matrix(labels, preds, class_names, output_dir, normalize=False)
    plot_confusion_matrix(labels, preds, class_names, output_dir, normalize=True)

    # Per-class F1
    plot_per_class_f1(metrics, class_names, output_dir)

    logger.info(f"Visualizations saved to: {output_dir}")

    print(f"\nEvaluation complete!")
    print(f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
