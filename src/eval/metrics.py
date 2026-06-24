"""Evaluation metrics for Conifer Family Classification."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@torch.no_grad()
def compute_metrics(
    model: nn.Module,
    data_loader,
    device: str = "cuda",
    num_classes: int = 4,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """Compute all evaluation metrics.

    Args:
        model: PyTorch model
        data_loader: Data loader for evaluation
        device: Device to use
        num_classes: Number of classes
        class_names: Optional list of class names

    Returns:
        Dictionary with all metrics
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    labels = list(range(num_classes))
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision_macro": precision_score(
            all_labels, all_preds, labels=labels, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            all_labels, all_preds, labels=labels, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            all_labels, all_preds, labels=labels, average="macro", zero_division=0
        ),
        "confusion_matrix": confusion_matrix(all_labels, all_preds, labels=labels),
    }

    # Per-class metrics
    per_class_f1 = f1_score(
        all_labels, all_preds, labels=labels, average=None, zero_division=0
    )
    per_class_precision = precision_score(
        all_labels, all_preds, labels=labels, average=None, zero_division=0
    )
    per_class_recall = recall_score(
        all_labels, all_preds, labels=labels, average=None, zero_division=0
    )

    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    metrics["per_class"] = {
        name: {
            "f1": float(per_class_f1[i]),
            "precision": float(per_class_precision[i]),
            "recall": float(per_class_recall[i]),
        }
        for i, name in enumerate(class_names)
    }

    return metrics


@torch.no_grad()
def measure_inference_time(
    model: nn.Module,
    input_size: Tuple[int, int, int] = (3, 224, 224),
    device: str = "cuda",
    num_runs: int = 100,
    warmup_runs: int = 10,
    batch_size: int = 1,
) -> Dict[str, float]:
    """Measure average inference time per image.

    Args:
        model: PyTorch model
        input_size: Input tensor size (C, H, W)
        device: Device to use
        num_runs: Number of timed runs
        warmup_runs: Number of warmup runs
        batch_size: Batch size for inference

    Returns:
        Dictionary with timing metrics
    """
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(batch_size, *input_size).to(device)

    # Warmup
    for _ in range(warmup_runs):
        _ = model(dummy_input)

    if device == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)

    times = []
    for _ in range(num_runs):
        start.record()
        _ = model(dummy_input)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "per_image_ms": float(np.mean(times) / batch_size),
    }


def format_metrics(metrics: Dict, class_names: Optional[List[str]] = None) -> str:
    """Format metrics dictionary as a readable string.

    Args:
        metrics: Dictionary of metrics
        class_names: Optional list of class names

    Returns:
        Formatted string
    """
    lines = []
    lines.append("=" * 50)
    lines.append("EVALUATION RESULTS")
    lines.append("=" * 50)
    lines.append(f"Accuracy:      {metrics['accuracy']*100:.2f}%")
    lines.append(f"Precision:     {metrics['precision_macro']*100:.2f}%")
    lines.append(f"Recall:        {metrics['recall_macro']*100:.2f}%")
    lines.append(f"F1-macro:      {metrics['f1_macro']*100:.2f}%")
    lines.append("")

    if "per_class" in metrics and class_names is not None:
        lines.append("Per-Class F1 Scores:")
        for name, scores in metrics["per_class"].items():
            lines.append(f"  {name:20s}: {scores['f1']*100:.2f}%")

    lines.append("=" * 50)

    return "\n".join(lines)
