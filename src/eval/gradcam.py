"""Grad-CAM visualization for Conifer Family Classification."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


class GradCAMGenerator:
    """Generate Grad-CAM visualizations for model interpretation.
    
    Args:
        model: PyTorch model
        backbone_name: Name of the backbone architecture
    """
    
    # Target layers for each backbone
    TARGET_LAYERS: Dict[str, str] = {
        "resnet50": "layer4[-1].conv3",
        "efficientnet_b3": "features[-1]",
        "efficientnet_b0": "features[-1]",
        "mobilenetv3_large": "features[-1]",
        "mobilenetv3_small": "features[-1]",
        "convnext_tiny": "features[-1]",
        "convnext_small": "features[-1]",
        "swin_tiny": "layers[-1].blocks[-1].norm1",
        "swin_small": "layers[-1].blocks[-1].norm1",
    }
    
    # ImageNet normalization
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    def __init__(self, model: nn.Module, backbone_name: str):
        self.model = model
        self.backbone_name = backbone_name
        self.target_layer = self._get_target_layer()
        self.cam = GradCAM(model=self.model, target_layers=[self.target_layer])
    
    def _get_target_layer(self) -> nn.Module:
        """Get the target layer for Grad-CAM based on backbone."""
        layer_name = self.TARGET_LAYERS.get(self.backbone_name, "layer4[-1].conv3")
        
        # Navigate to the target layer
        parts = layer_name.replace("[-1]", "").split(".")
        layer = self.model
        
        for part in parts:
            if hasattr(layer, part):
                layer = getattr(layer, part)
            else:
                # Try to find it in submodules
                for name, module in layer.named_modules():
                    if name.endswith(part):
                        layer = module
                        break
        
        return layer
    
    @torch.no_grad()
    def generate(
        self,
        image: torch.Tensor,
        target_category: Optional[int] = None,
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap for an image.
        
        Args:
            image: (C, H, W) tensor, normalized
            target_category: Target class index (None for highest scoring)
        
        Returns:
            (H, W, 3) uint8 image with heatmap overlay
        """
        # Generate CAM
        grayscale_cam = self.cam(
            input_tensor=image.unsqueeze(0),
            target_category=target_category,
        )
        grayscale_cam = grayscale_cam[0, :]
        
        # Denormalize image
        mean = torch.tensor(self.IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(self.IMAGENET_STD).view(3, 1, 1)
        img_denorm = image * std + mean
        img_np = img_denorm.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)
        
        # Overlay heatmap
        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        
        return visualization
    
    def generate_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        output_dir: Path,
        class_names: List[str],
        num_samples: int = 4,
    ) -> None:
        """Generate Grad-CAM visualizations for a batch of images.
        
        Args:
            images: Batch of images (B, C, H, W)
            labels: Ground truth labels (B,)
            output_dir: Directory to save visualizations
            class_names: List of class names
            num_samples: Number of samples to visualize per class
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for class_idx, class_name in enumerate(class_names):
            class_dir = output_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            # Get indices for this class
            class_indices = (labels == class_idx).nonzero(as_tuple=True)[0]
            class_indices = class_indices[:num_samples]
            
            for idx in class_indices:
                img = images[idx]
                cam = self.generate(img)
                
                # Save
                save_path = class_dir / f"sample_{idx.item()}.png"
                plt.imsave(save_path, cam)
    
    def generate_comparison(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        class_names: List[str],
        num_samples: int = 4,
    ) -> plt.Figure:
        """Generate comparison figure with original and Grad-CAM images.
        
        Args:
            images: Batch of images (B, C, H, W)
            labels: Ground truth labels (B,)
            class_names: List of class names
            num_samples: Number of samples to visualize
        
        Returns:
            Matplotlib figure
        """
        num_samples = min(num_samples, len(images))
        
        fig, axes = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8))
        
        for i in range(num_samples):
            img = images[i]
            label = labels[i].item()
            
            # Original image
            mean = torch.tensor(self.IMAGENET_MEAN).view(3, 1, 1)
            std = torch.tensor(self.IMAGENET_STD).view(3, 1, 1)
            img_denorm = img * std + mean
            img_np = img_denorm.permute(1, 2, 0).numpy()
            img_np = np.clip(img_np, 0, 1)
            
            axes[0, i].imshow(img_np)
            axes[0, i].set_title(f"True: {class_names[label]}")
            axes[0, i].axis("off")
            
            # Grad-CAM
            cam = self.generate(img)
            axes[1, i].imshow(cam)
            axes[1, i].set_title("Grad-CAM")
            axes[1, i].axis("off")
        
        plt.tight_layout()
        
        return fig
