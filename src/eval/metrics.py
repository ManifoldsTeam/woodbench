"""Shared evaluation metrics for WoodBench classification tasks.

The current Task B benchmark has exactly 17 species classes.  Every Task B
backbone and seed must use :func:`compute_task_b_metrics` so the metric
definition and label universe cannot drift between notebooks.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

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

TASK_B_NUM_CLASSES = 17
TASK_B_LABELS = tuple(range(TASK_B_NUM_CLASSES))


def _compute_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    *,
    labels: Sequence[int],
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Compute classification metrics over an explicit label universe."""
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    fixed_labels = list(labels)

    if true.ndim != 1 or pred.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional")
    if true.size == 0:
        raise ValueError("y_true and y_pred must not be empty")
    if true.shape[0] != pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")
    if not fixed_labels:
        raise ValueError("labels must not be empty")
    if len(set(fixed_labels)) != len(fixed_labels):
        raise ValueError("labels must contain unique class indices")

    declared = set(fixed_labels)
    observed = set(true.tolist()) | set(pred.tolist())
    unknown = observed - declared
    if unknown:
        raise ValueError(
            f"y_true/y_pred contain class indices outside labels: {sorted(unknown)}"
        )

    if class_names is None:
        fixed_class_names = [f"class_{label}" for label in fixed_labels]
    else:
        fixed_class_names = list(class_names)
        if len(fixed_class_names) != len(fixed_labels):
            raise ValueError("class_names must have the same length as labels")
        if len(set(fixed_class_names)) != len(fixed_class_names):
            raise ValueError("class_names must be unique")

    per_class_precision = precision_score(
        true, pred, labels=fixed_labels, average=None, zero_division=0
    )
    per_class_recall = recall_score(
        true, pred, labels=fixed_labels, average=None, zero_division=0
    )
    per_class_f1 = f1_score(
        true, pred, labels=fixed_labels, average=None, zero_division=0
    )

    return {
        "accuracy": float(accuracy_score(true, pred)),
        "precision_macro": float(
            precision_score(
                true,
                pred,
                labels=fixed_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                true,
                pred,
                labels=fixed_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                true,
                pred,
                labels=fixed_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            true, pred, labels=fixed_labels
        ),
        "per_class": {
            name: {
                "label": label,
                "precision": float(per_class_precision[index]),
                "recall": float(per_class_recall[index]),
                "f1": float(per_class_f1[index]),
            }
            for index, (label, name) in enumerate(
                zip(fixed_labels, fixed_class_names)
            )
        },
        "labels": fixed_labels,
        "class_names": fixed_class_names,
        "average": "macro",
        "zero_division": 0,
    }


def compute_task_b_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    *,
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Compute the canonical metrics for the current 17-class Task B.

    Macro precision, recall, and F1 always use labels ``0..16`` and
    ``zero_division=0``. Callers cannot supply another label universe. If
    ``class_names`` is provided, it must contain exactly 17 unique names in
    checkpoint-output order.
    """
    return _compute_classification_metrics(
        y_true,
        y_pred,
        labels=TASK_B_LABELS,
        class_names=class_names,
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
    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    return _compute_classification_metrics(
        all_labels,
        all_preds,
        labels=list(range(num_classes)),
        class_names=class_names,
    )


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
