"""Optimizer and scheduler factories for Conifer Family Classification.

Supports differential learning rates for fine-tuning: the backbone uses a
lower learning rate (typically 1/10th of the head LR) to preserve pretrained
features while the classifier head learns task-specific patterns.
"""

import torch
import torch.nn as nn

from .loss import LabelSmoothingCrossEntropy


def create_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    backbone_lr: float | None = None,
) -> torch.optim.Optimizer:
    """Create AdamW optimizer with optional differential learning rates.

    When backbone_lr is provided, the optimizer uses two parameter groups:
    - "head": learning rate = lr (higher, task-specific)
    - "backbone": learning rate = backbone_lr (lower, pretrained features)

    When backbone_lr is None, all parameters use the same learning rate.

    Args:
        model: PyTorch model (must have .named_parameters())
        lr: Learning rate for the classifier head
        weight_decay: Weight decay
        backbone_lr: Optional lower learning rate for the backbone.
            If None, all parameters use `lr`. Recommended: lr / 10 (e.g., 1e-5).

    Returns:
        AdamW optimizer with param groups
    """
    if backbone_lr is not None and backbone_lr != lr:
        head_params = []
        backbone_params = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if "head" in name or "fc" in name or "classifier" in name:
                    head_params.append(param)
                else:
                    backbone_params.append(param)

        # Fallback: if no head params found, use all non-head params as backbone
        # and everything as head (for models where timm uses different naming)
        if not head_params:
            head_params = list(model.parameters())
            backbone_params = []
        elif not backbone_params:
            backbone_params = [p for n, p in model.named_parameters()
                               if p not in head_params]

        param_groups = [
            {"params": head_params, "lr": lr, "name": "head"},
            {"params": backbone_params, "lr": backbone_lr, "name": "backbone"},
        ]
        return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    else:
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int = 50,
    scheduler_type: str = "cosine",
    warmup_epochs: int = 5,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler with optional linear warmup.

    When warmup_epochs > 0, a linear warmup is applied before the main
    scheduler (cosine/step). The warmup ramps LR from 0 to its initial value
    over `warmup_epochs`, then the main scheduler takes over.

    Args:
        optimizer: PyTorch optimizer
        epochs: Number of training epochs
        scheduler_type: 'cosine', 'step', or 'plateau'
        warmup_epochs: Number of warmup epochs (0 = no warmup)

    Returns:
        Learning rate scheduler (possibly wrapped in SequentialLR)
    """
    if warmup_epochs > 0 and scheduler_type in ("cosine", "step"):
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        if scheduler_type == "cosine":
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs - warmup_epochs,
            )
        else:
            main_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=20,
                gamma=0.1,
            )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs],
        )

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
