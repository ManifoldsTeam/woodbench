"""Model factory for Conifer Family Classification.

Creates pretrained backbones with classification heads.
"""

from typing import Dict

import timm
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
    freeze_backbone: bool = True,
) -> nn.Module:
    """Create model with pretrained backbone and classification head.
    
    Args:
        backbone_name: Model name - either friendly name (e.g., "resnet50") 
                       or full timm name (e.g., "resnet50", "tf_efficientnet_b3")
        num_classes: Number of output classes (4 for Task A family classification)
        pretrained: Use ImageNet pretrained weights
        freeze_backbone: Freeze backbone parameters (only train head)
    
    Returns:
        Model with backbone + classification head
    
    Example:
        >>> model = create_model("resnet50", num_classes=4)
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
    
    # Create new classification head
    model.head = nn.Linear(in_features, num_classes)
    
    # Freeze backbone if specified
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        # Ensure head is trainable
        for param in model.head.parameters():
            param.requires_grad = True
    
    return model


def get_model_info(backbone_name: str) -> Dict:
    """Get model information including parameters and FLOPs.
    
    Args:
        backbone_name: Model name
    
    Returns:
        Dictionary with model info
    """
    model = create_model(backbone_name, num_classes=4, pretrained=False)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "backbone": backbone_name,
        "timm_name": BACKBONE_NAMES.get(backbone_name, backbone_name),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
    }
