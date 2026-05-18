"""Profiling entry point for WoodID-S2-2026.

Measure efficiency metrics for all backbone models:
    python profile.py
    python profile.py model.backbone=resnet50
"""

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src import create_model, setup_logger
from src.profile import profile_model


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Profile model efficiency."""
    print(OmegaConf.to_yaml(cfg))

    logger = setup_logger()
    model_cfg = cfg.get("model", {})
    num_classes = model_cfg.get("num_classes", 4)
    backbone = model_cfg.get("backbone", "resnet50")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("GPU not available. Profiling on CPU.")

    logger.info(f"Profiling {backbone}...")

    result = profile_model(
        backbone_name=backbone,
        num_classes=num_classes,
        input_size=(3, 224, 224),
        device=device,
        num_runs=100,
    )

    logger.info("=" * 60)
    logger.info(f"Model: {backbone}")
    logger.info("=" * 60)
    logger.info(f"  Total parameters:  {result['total_millions']:.2f}M")
    logger.info(f"  Trainable params:  {result['trainable_millions']:.2f}M")
    logger.info(f"  Model size:        {result['size_mb']:.2f} MB")
    if result.get("flops_giga") is not None:
        logger.info(f"  FLOPs:             {result['flops_giga']:.2f} G")
    logger.info(f"  Inference time:    {result['inference_time_ms']:.2f} ms (batch=1)")
    logger.info(f"  Per-image time:    {result['inference_time_per_image_ms']:.2f} ms")

    output_path = Path("logs/profiling") / f"{backbone}_profile.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import json
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Profile saved: {output_path}")


if __name__ == "__main__":
    main()
