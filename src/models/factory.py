"""Model factory for Conifer Family Classification.

Creates pretrained backbones with classification heads.
Supports both frozen-backbone (linear probing) and full fine-tuning.

Training Strategy (per WoodID_plan.md):
    Fine-tuning with unfrozen backbone + differential learning rates
    (backbone_lr=1e-5, head_lr=1e-4) is the recommended approach.
    This adapts pretrained ImageNet features to the wood cross-section
    domain while preserving low-level texture features.
"""

from typing import Dict

import timm
import torch
import torch.nn as nn


# Backbone name mapping: friendly name -> timm model name
BACKBONE_NAMES: Dict[str, str] = {
    "resnet50": "resnet50",
    "efficientnet_b3": "tf_efficientnet_b3",
    "mobilenetv3_large": "mobilenetv3_large_100",
    "convnext_tiny": "convnext_tiny",
    "swin_tiny": "swin_tiny_patch4_window7_224",
}


def create_model(
    backbone_name: str,
    num_classes: int = 4,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Create model with pretrained backbone and classification head.

    The model is built from a timm pretrained backbone with the original
    classifier replaced by a new linear head.

    Args:
        backbone_name: Model name — either friendly name (e.g., "resnet50")
            or full timm name (e.g., "tf_efficientnet_b3")
        num_classes: Number of output classes (4 for Task A family, 20 for Task B species)
        pretrained: Use ImageNet pretrained weights
        freeze_backbone: If True, freeze all backbone parameters (linear probing).
            If False (default), all parameters are trainable (fine-tuning).
            Fine-tuning with differential LR is recommended per WoodID_plan.md.

    Returns:
        Model with backbone + classification head

    Example:
        >>> model = create_model("resnet50", num_classes=4, freeze_backbone=False)
        >>> print(model)
    """
    # Resolve backbone name
    if backbone_name in BACKBONE_NAMES:
        timm_name = BACKBONE_NAMES[backbone_name]
    else:
        timm_name = backbone_name

    # Create model with the correct number of output classes.
    # timm includes the classification head in the forward path when num_classes > 0.
    # We then replace it with our own linear head via reset_classifier().
    model = timm.create_model(
        timm_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )

    # Get feature dimension
    in_features = model.num_features

    # Replace the classifier with a simple linear head.
    # timm.reset_classifier() correctly replaces the final classification layer
    # regardless of the internal attribute name (fc / head / classifier).
    model.reset_classifier(num_classes=num_classes)

    # Verify output shape
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy_input)
        assert out.shape == (1, num_classes), (
            f"Expected output shape (1, {num_classes}), got {out.shape}"
        )

    # Freeze backbone if specified (linear probing mode)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze the final classifier — find the Linear with out_features == num_classes
        for module in model.modules():
            if isinstance(module, nn.Linear) and module.out_features == num_classes:
                for param in module.parameters():
                    param.requires_grad = True
                break

    return model


def get_model_info(backbone_name: str, freeze_backbone: bool = False) -> Dict:
    """Get model information including parameter counts.

    Args:
        backbone_name: Model name
        freeze_backbone: Whether backbone is frozen (affects trainable count).
            Default False (fine-tuning).

    Returns:
        Dictionary with model info
    """
    model = create_model(backbone_name, num_classes=4, pretrained=False, freeze_backbone=freeze_backbone)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "backbone": backbone_name,
        "timm_name": BACKBONE_NAMES.get(backbone_name, backbone_name),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
        "mode": "fine-tuning" if not freeze_backbone else "linear_probing",
    }
