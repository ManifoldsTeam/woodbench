"""Checkpoint utilities for model loading and saving.

Provides dataclass-based checkpoint structure matching the design doc.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch


@dataclass
class TrainingMetadata:
    """Training metadata stored in checkpoint files."""
    backbone: str
    task: str
    num_classes: int
    epochs: int
    learning_rate: float
    weight_decay: float
    label_smoothing: float
    seed: int
    device: str
    family_classes: list = field(default_factory=list)
    best_val_acc: float = 0.0
    best_epoch: int = 0
    timestamp: str = ""


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load model checkpoint with full metadata."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
        "best_val_acc": checkpoint.get("best_val_acc", checkpoint.get("best_value", 0.0)),
        "best_epoch": checkpoint.get("best_epoch", 0),
    }


def save_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    best_val_acc: float = 0.0,
    best_epoch: int = 0,
    scheduler: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model checkpoint with metadata."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if metadata is not None:
        checkpoint["metadata"] = metadata
    torch.save(checkpoint, checkpoint_path)
