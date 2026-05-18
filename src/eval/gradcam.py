"""Grad-CAM visualization for Conifer Family Classification.

Generates Class Activation Maps using torchcam library for model interpretation.
Supports per-backbone target layers as specified in the design doc.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchcam.methods import GradCAM, LayerCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image


class GradCAMGenerator:
    """Generate Grad-CAM visualizations for model interpretation.

    Args:
        model: PyTorch model
        backbone_name: Name of the backbone architecture
    """

    # Target layers for each backbone (per design doc)
    TARGET_LAYERS: Dict[str, str] = {
        "resnet50": "layer4.1.conv3",
        "efficientnet_b3": "features.8",
        "efficientnet_b0": "features.6",
        "mobilenetv3_large": "features.17",
        "mobilenetv3_small": "features.11",
        "convnext_tiny": "features.7",
        "convnext_small": "features.7",
        "swin_tiny": "head",
        "swin_small": "head",
    }

    # ImageNet normalization
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, model: nn.Module, backbone_name: str, method: str = "gradcam"):
        self.model = model
        self.backbone_name = backbone_name
        self.method = method
        self.target_layer = self._get_target_layer()
        self.cam_extractor: Optional[LayerCAM] = None

    def _get_target_layer(self) -> nn.Module:
        """Get the target layer for Grad-CAM based on backbone."""
        layer_name = self.TARGET_LAYERS.get(self.backbone_name, "layer4.1.conv3")

        layer = self.model
        for part in layer_name.split("."):
            if hasattr(layer, part):
                layer = getattr(layer, part)
            else:
                for name, module in layer.named_modules():
                    if name.endswith(part):
                        layer = module
                        break
                break

        return layer

    def _get_cam_extractor(self) -> LayerCAM:
        """Get or create the CAM extractor."""
        if self.cam_extractor is None:
            if self.method == "layer":
                self.cam_extractor = LayerCAM(self.target_layer)
            else:
                self.cam_extractor = GradCAM(self.target_layer)
        return self.cam_extractor

    @torch.no_grad()
    def generate(
        self,
        image: torch.Tensor,
        target_category: Optional[int] = None,
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap for an image."""
        cam_extractor = self._get_cam_extractor()

        input_tensor = image.unsqueeze(0)
        out = self.model(input_tensor)

        if target_category is None:
            class_idx = out.squeeze(0).argmax().item()
        else:
            class_idx = target_category

        activation_map = cam_extractor(class_idx, out)
        activation_map = activation_map[0].squeeze(0).cpu().numpy()

        img_denorm = self._denormalize(image)
        img_np = img_denorm.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)

        result = overlay_mask(
            to_pil_image(img_np),
            to_pil_image(activation_map, mode="F"),
            alpha=0.5,
        )

        return np.array(result)

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

            class_indices = (labels == class_idx).nonzero(as_tuple=True)[0]
            class_indices = class_indices[:num_samples]

            for idx in class_indices:
                img = images[idx]
                cam = self.generate(img)
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

            img_denorm = self._denormalize(img)
            img_np = img_denorm.permute(1, 2, 0).numpy()
            img_np = np.clip(img_np, 0, 1)

            axes[0, i].imshow(img_np)
            axes[0, i].set_title(f"True: {class_names[label]}")
            axes[0, i].axis("off")

            cam = self.generate(img)
            axes[1, i].imshow(cam)
            axes[1, i].set_title("Grad-CAM")
            axes[1, i].axis("off")

        plt.tight_layout()

        return fig

    def _denormalize(self, image: torch.Tensor) -> torch.Tensor:
        """Denormalize an image using ImageNet mean/std."""
        mean = torch.tensor(self.IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(self.IMAGENET_STD).view(3, 1, 1)
        return image * std + mean

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *args):
        """Context manager exit — clean up CAM extractor hooks."""
        if self.cam_extractor is not None:
            self.cam_extractor.disable_hooks()
        return False
