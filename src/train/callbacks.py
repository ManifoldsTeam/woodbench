"""Training callbacks for Conifer Family Classification."""

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn


class CheckpointCallback:
    """Callback for saving model checkpoints during training.
    
    Args:
        output_dir: Directory to save checkpoints
        name: Base name for checkpoint files
        save_best_only: Only save best model
        metric: Metric to monitor for best model
        mode: 'max' or 'min' for metric comparison
    """
    
    def __init__(
        self,
        output_dir: Path,
        name: str = "model",
        save_best_only: bool = True,
        metric: str = "val_acc",
        mode: str = "max",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.save_best_only = save_best_only
        self.metric = metric
        self.mode = mode
        
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.best_epoch = 0
    
    def __call__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> bool:
        """Check if model should be saved.
        
        Returns:
            True if checkpoint was saved
        """
        metric_value = metrics.get(self.metric, None)
        
        if metric_value is None:
            return False
        
        is_best = (
            (self.mode == "max" and metric_value > self.best_value) or
            (self.mode == "min" and metric_value < self.best_value)
        )
        
        if is_best:
            self.best_value = metric_value
            self.best_epoch = epoch
        
        if is_best or not self.save_best_only:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "best_value": self.best_value,
                "best_epoch": self.best_epoch,
            }
            
            if scheduler is not None:
                checkpoint["scheduler_state_dict"] = scheduler.state_dict()
            
            if is_best:
                save_path = self.output_dir / f"{self.name}_best.pth"
                torch.save(checkpoint, save_path)
                return True
        
        return False


class EarlyStoppingCallback:
    """Callback for early stopping based on validation metric.
    
    Args:
        metric: Metric to monitor
        mode: 'max' or 'min'
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
    """
    
    def __init__(
        self,
        metric: str = "val_acc",
        mode: str = "max",
        patience: int = 10,
        min_delta: float = 0.0,
    ):
        self.metric = metric
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.should_stop = False
    
    def __call__(self, metrics: Dict) -> bool:
        """Check if training should stop.
        
        Returns:
            True if training should stop
        """
        metric_value = metrics.get(self.metric, None)
        
        if metric_value is None:
            return False
        
        if self.mode == "max":
            improved = metric_value > self.best_value + self.min_delta
        else:
            improved = metric_value < self.best_value - self.min_delta
        
        if improved:
            self.best_value = metric_value
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.should_stop = True
            return True
        
        return False
    
    def reset(self):
        """Reset the early stopping state."""
        self.counter = 0
        self.should_stop = False
