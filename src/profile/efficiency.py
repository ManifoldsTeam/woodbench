"""Efficiency profiling for Conifer Family Classification."""

from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

from ..models.factory import BACKBONE_NAMES, create_model


def count_parameters(model: nn.Module) -> Dict[str, float]:
    """Count model parameters.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "total_millions": total / 1e6,
        "trainable_millions": trainable / 1e6,
    }


def measure_model_size(model: nn.Module) -> Dict[str, float]:
    """Measure model file size.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with size information
    """
    import tempfile
    import os
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as f:
        torch.save(model.state_dict(), f.name)
        size_bytes = os.path.getsize(f.name)
        size_mb = size_bytes / (1024 * 1024)
        os.unlink(f.name)
    
    return {
        "size_bytes": size_bytes,
        "size_mb": size_mb,
    }


def profile_model(
    backbone_name: str,
    num_classes: int = 4,
    input_size: tuple = (3, 224, 224),
    device: str = "cuda",
    num_runs: int = 100,
    batch_size: int = 1,
) -> Dict:
    """Profile a model for efficiency metrics.
    
    Args:
        backbone_name: Model name
        num_classes: Number of output classes
        input_size: Input tensor size (C, H, W)
        device: Device to use
        num_runs: Number of inference runs for timing
        batch_size: Batch size for inference
    
    Returns:
        Dictionary with all profiling metrics
    """
    # Create model
    model = create_model(backbone_name, num_classes=num_classes, pretrained=False)
    model = model.to(device)
    model.eval()
    
    # Count parameters
    params = count_parameters(model)
    
    # Measure model size
    size = measure_model_size(model)
    
    # Measure FLOPs (if available)
    flops = {"flops_giga": None}
    try:
        from thop import profile
        dummy_input = torch.randn(1, *input_size).to(device)
        gflops, _ = profile(model, inputs=(dummy_input,), verbose=False)
        flops = {"flops_giga": gflops / 1e9}
    except ImportError:
        try:
            from ptflops import get_model_complexity_info
            macs, _ = get_model_complexity_info(
                model, input_size, verbose=False
            )
            flops = {"flops_giga": macs / 2 / 1e9}  # GFLOPs = 2 * GMACs
        except ImportError:
            pass
    
    # Measure inference time
    inference_time = measure_inference_time_gpu(
        model, input_size, device, num_runs, batch_size
    )
    
    return {
        "backbone": backbone_name,
        **params,
        **size,
        **flops,
        **inference_time,
    }


def measure_inference_time_gpu(
    model: nn.Module,
    input_size: tuple,
    device: str,
    num_runs: int = 100,
    batch_size: int = 1,
) -> Dict[str, float]:
    """Measure inference time on GPU.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (C, H, W)
        device: Device to use
        num_runs: Number of runs
        batch_size: Batch size
    
    Returns:
        Dictionary with timing metrics
    """
    import numpy as np
    
    model.eval()
    dummy_input = torch.randn(batch_size, *input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = torch.Event(enable_timing=True)
            end = torch.Event(enable_timing=True)

            start.record()
            _ = model(dummy_input)
            end.record()
            torch.cuda.synchronize()

            times.append(start.elapsed_time(end))
    
    mean_ms = np.mean(times)
    
    return {
        "inference_time_ms": mean_ms,
        "inference_time_per_image_ms": mean_ms / batch_size,
        "inference_time_std_ms": np.std(times),
    }


def profile_all_models(
    output_path: Path = None,
    num_runs: int = 100,
) -> List[Dict]:
    """Profile all backbone models.
    
    Args:
        output_path: Optional path to save results
        num_runs: Number of inference runs
    
    Returns:
        List of profiling results for each model
    """
    results = []
    
    for friendly_name in BACKBONE_NAMES.keys():
        print(f"Profiling {friendly_name}...")
        result = profile_model(friendly_name, num_runs=num_runs)
        results.append(result)
        
        print(f"  Params: {result['total_millions']:.2f}M")
        print(f"  Size: {result['size_mb']:.2f} MB")
        if result['flops_giga']:
            print(f"  FLOPs: {result['flops_giga']:.2f} G")
        print(f"  Inference: {result['inference_time_per_image_ms']:.2f} ms/image")
    
    if output_path is not None:
        import json
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
    
    return results
