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

    # Create model with no classifier (num_classes=0)
    model = timm.create_model(
        timm_name,
        pretrained=pretrained,
        num_classes=0,  # Remove default classifier
    )

    # Get feature dimension from backbone
    in_features = model.num_features

    # Create new classification head.
    # IMPORTANT: Different timm architectures use different forward paths:
    # - ResNet, EfficientNet, MobileNet: forward calls model.head
    # - ConvNeXt: forward calls model.head.fc (model.head is a Sequential)
    # - Swin Transformer: forward calls model.head (model.head is a Sequential)
    # So we must attach the head at the correct attribute for each architecture.
    if "convnext" in timm_name:
        # ConvNeXt: head is a Sequential([norm, flatten, fc])
        # Replace just the final fc layer to keep the norm/flatten chain
        model.head.fc = nn.Linear(in_features, num_classes)
    elif "swin" in timm_name:
        # Swin-T: head is a Sequential([flatten, norm, fc])
        # Replace the final fc layer
        model.head.fc = nn.Linear(in_features, num_classes)
    else:
        # ResNet, EfficientNet, MobileNet: head is just nn.Linear
        model.head = nn.Linear(in_features, num_classes)

    # Verify the head is in the forward path with a quick forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        try:
            out = model(dummy_input)
            assert out.shape[-1] == num_classes, (
                f"Model output shape mismatch: expected ({num_classes},), got {out.shape}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Model forward pass failed for '{backbone_name}' ({timm_name}). "
                f"Check that the classification head is correctly attached. Error: {e}"
            ) from e

    # Freeze backbone if specified (linear probing mode)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        # Ensure head is trainable (unfreeze it)
        for module in model.head.modules():
            for param in module.parameters():
                param.requires_grad = True

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
