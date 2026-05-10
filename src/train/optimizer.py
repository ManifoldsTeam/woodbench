"""Optimizer and scheduler factories for Conifer Family Classification."""

import torch
import torch.nn as nn


def create_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """Create AdamW optimizer.
    
    Args:
        model: PyTorch model
        lr: Learning rate
        weight_decay: Weight decay
    
    Returns:
        AdamW optimizer
    """
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int = 50,
    scheduler_type: str = "cosine",
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        epochs: Number of training epochs
        scheduler_type: 'cosine', 'step', or 'plateau'
    
    Returns:
        Learning rate scheduler
    """
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
        )
    elif scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=20,
            gamma=0.1,
        )
    elif scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.1,
            patience=5,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing.
    
    Args:
        smoothing: Smoothing factor (0.0 to 1.0)
    """
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate loss.
        
        Args:
            pred: Model predictions (logits)
            target: Ground truth labels
        
        Returns:
            Loss value
        """
        num_classes = pred.size(1)
        
        log_probs = torch.nn.functional.log_softmax(pred, dim=1)
        
        # One-hot encode targets
        one_hot = torch.zeros_like(log_probs).scatter_(1, target.unsqueeze(1), 1)
        
        # Apply label smoothing
        smoothed_targets = one_hot * self.confidence + self.smoothing / num_classes
        
        loss = (-smoothed_targets * log_probs).sum(dim=1).mean()
        
        return loss


def create_criterion(
    criterion_type: str = "crossentropy",
    label_smoothing: float = 0.1,
) -> nn.Module:
    """Create loss function.
    
    Args:
        criterion_type: 'crossentropy', 'label_smoothing', or 'crossentropy_label_smoothing'
        label_smoothing: Label smoothing factor (only for label smoothing criterion)
    
    Returns:
        Loss function
    """
    if criterion_type == "crossentropy":
        return nn.CrossEntropyLoss()
    elif criterion_type in ("label_smoothing", "crossentropy_label_smoothing"):
        return LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    else:
        raise ValueError(f"Unknown criterion type: {criterion_type}")
