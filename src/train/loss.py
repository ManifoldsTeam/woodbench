"""Loss functions for Conifer Family Classification."""

import torch
import torch.nn as nn


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing.

    Reduces overconfidence by distributing a small portion of probability mass
    uniformly across all classes. Standard regularization for classification.

    Args:
        smoothing: Smoothing factor in [0.0, 1.0]. 0.0 = standard CE, 0.1 = 10% smoothing.
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate loss.

        Args:
            pred: Model predictions (logits), shape (batch, num_classes)
            target: Ground truth labels, shape (batch,)

        Returns:
            Scalar loss value
        """
        num_classes = pred.size(1)
        log_probs = torch.nn.functional.log_softmax(pred, dim=1)
        one_hot = torch.zeros_like(log_probs).scatter_(1, target.unsqueeze(1), 1)
        smoothed_targets = one_hot * self.confidence + self.smoothing / num_classes
        loss = (-smoothed_targets * log_probs).sum(dim=1).mean()
        return loss


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.

    Down-weights well-classified examples, focusing training on hard samples.

    Args:
        alpha: Weighting factor per class
        gamma: Focusing parameter (higher = more focus on hard samples)
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = torch.nn.functional.cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def create_loss_function(
    loss_type: str = "crossentropy",
    label_smoothing: float = 0.1,
    **kwargs,
) -> nn.Module:
    """Factory function to create loss functions.

    Args:
        loss_type: Type of loss — "crossentropy", "label_smoothing", "focal"
        label_smoothing: Label smoothing factor (for label_smoothing loss)
        **kwargs: Additional arguments passed to loss constructor

    Returns:
        Loss function module
    """
    if loss_type == "crossentropy":
        return nn.CrossEntropyLoss()
    elif loss_type == "label_smoothing":
        return LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    elif loss_type == "focal":
        return FocalLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
