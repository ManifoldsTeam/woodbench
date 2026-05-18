"""Benchmark utilities for comparing multiple models.

Runs efficiency profiling and evaluation across all configured backbones,
generating comparison tables and reports.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from ..models.factory import BACKBONE_NAMES, create_model
from ..profile.efficiency import profile_model


@dataclass
class BenchmarkConfig:
    """Configuration for model benchmarking."""
    num_classes: int = 4
    input_size: tuple = (3, 224, 224)
    device: str = "cuda"
    num_runs: int = 100


def benchmark_all_models(
    config: BenchmarkConfig,
    output_path: Optional[Path] = None,
) -> List[Dict]:
    """Profile all configured backbone models."""
    results = []
    for backbone in BACKBONE_NAMES:
        result = profile_model(
            backbone_name=backbone,
            num_classes=config.num_classes,
            input_size=config.input_size,
            device=config.device,
            num_runs=config.num_runs,
        )
        results.append(result)
        _log_result(result)
    if output_path:
        _save_results(results, output_path)
    return results


def compare_models(
    results: List[Dict],
    sort_by: str = "accuracy",
    higher_is_better: bool = True,
) -> List[Dict]:
    """Sort and compare model results."""
    return sorted(
        results,
        key=lambda x: x.get(sort_by, 0),
        reverse=higher_is_better,
    )


def print_comparison_table(results: List[Dict]) -> None:
    """Print a formatted comparison table."""
    header = f"{'Backbone':<20} {'Params(M)':>10} {'Size(MB)':>10} {'FLOPs(G)':>10} {'Time(ms)':>10}"
    print("=" * 70)
    print(header)
    print("=" * 70)
    for r in results:
        flops = f"{r.get('flops_giga', 0):.2f}" if r.get("flops_giga") else "N/A"
        print(
            f"{r['backbone']:<20} "
            f"{r.get('total_millions', 0):>10.2f} "
            f"{r.get('size_mb', 0):>10.2f} "
            f"{flops:>10} "
            f"{r.get('inference_time_ms', 0):>10.2f}"
        )
    print("=" * 70)


def _log_result(result: Dict) -> None:
    """Log a profiling result."""
    flops_str = f"{result['flops_giga']:.2f} G" if result.get("flops_giga") else "N/A"
    print(
        f"  Params: {result['total_millions']:.2f}M | "
        f"Size: {result['size_mb']:.2f} MB | "
        f"FLOPs: {flops_str} | "
        f"Time: {result['inference_time_ms']:.2f} ms"
    )


def _save_results(results: List[Dict], output_path: Path) -> None:
    """Save benchmark results to JSON."""
    import json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {output_path}")
