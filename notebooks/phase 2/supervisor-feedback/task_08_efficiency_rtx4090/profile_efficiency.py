#!/usr/bin/env python3
"""Step 5: latency/VRAM of the LR-selected checkpoints on the GPU at hand.

Same protocol as phase2_benchmark.run_efficiency_profile (batch 1, 3x224x224,
FP32, 50 warmup + 200 timed runs with torch.cuda.Event, peak allocated VRAM,
THOP MACs), reusing its measure_latency/live_environment, but without the
RTX 3060 guard and without writing into the WoodID repo.

For every architecture the selected step-1 checkpoint (backbone LR 1e-4,
seed 42, Final-B split) is timed, and so is the same architecture with random
weights. Equal numbers show that latency/VRAM depend on the architecture, not
on the trained weights, so a change of LR does not change them on a given GPU.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks/phase 2"))

import numpy as np
import pandas as pd
import torch
from thop import profile as thop_profile

from phase2_benchmark import (
    BACKBONES,
    DISPLAY_NAMES,
    live_environment,
    measure_latency,
    state_dict_from_checkpoint,
)
from src.models import create_model

STEP1_ROOT = ROOT / "data/experiments/Task-B-lr-select/bblr_1e-4/seed_42"
CV_SUMMARY_GLOB = "data/experiments/Task-B-origin-cv-v2/reports/cv-*/summary.csv"


def time_model(model, warmup, repetitions):
    model = model.cuda().eval()
    dummy = torch.randn(1, 3, 224, 224, device="cuda")
    macs, _ = thop_profile(model, inputs=(dummy,), verbose=False)
    # thop registers buffers/hooks; rebuild timing on a clean forward path.
    samples, baseline, peak = measure_latency(model, dummy, warmup, repetitions)
    del dummy
    return samples, baseline, peak, macs


def summarize(samples):
    return {
        "latency_mean_ms": float(samples.mean()),
        "latency_median_ms": float(np.median(samples)),
        "latency_sd_ms": float(samples.std(ddof=1)),
        "latency_p95_ms": float(np.percentile(samples, 95)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--threads", type=int, default=8, help="torch CPU threads")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "artifacts/phase_2/supervisor-feedback/task_08_efficiency_rtx4090",
    )
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.backends.cudnn.benchmark = False
    # The container sees 255 cores but has a ~24-core CPU quota; PyTorch's
    # default thread count then oversubscribes the CPU (model init took ~40 s).
    torch.set_num_threads(args.threads)
    environment = live_environment()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.output_dir / f"profile-{stamp}"
    out.mkdir(parents=True, exist_ok=False)

    cv = pd.read_csv(sorted(ROOT.glob(CV_SUMMARY_GLOB))[-1]).set_index("backbone")
    rows, sample_rows = [], []
    for backbone in BACKBONES:
        run = STEP1_ROOT / backbone
        metrics = json.loads((run / "metrics.json").read_text())
        checkpoint = run / "model_best.pth"
        for weights in ("checkpoint", "random_init"):
            torch.manual_seed(0)
            model = create_model(backbone, num_classes=17, pretrained=False)
            if weights == "checkpoint":
                result = model.load_state_dict(state_dict_from_checkpoint(checkpoint), strict=True)
                if result.missing_keys or result.unexpected_keys:
                    raise RuntimeError(f"State-dict mismatch for {backbone}: {result}")
            params = sum(p.numel() for p in model.parameters())
            state_bytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())
            print(f"Profiling {DISPLAY_NAMES[backbone]} ({weights}) on {environment['gpu']} ...")
            samples, baseline, peak, macs = time_model(model, args.warmup, args.repetitions)
            rows.append(
                {
                    "backbone": backbone,
                    "display_name": DISPLAY_NAMES[backbone],
                    "weights": weights,
                    "gpu": environment["gpu"],
                    "batch_size": 1,
                    "image_size": 224,
                    "precision": "FP32",
                    "total_params": int(params),
                    "gmacs": float(macs / 1e9),
                    "state_dict_size_mib": float(state_bytes / 1024**2),
                    **summarize(samples),
                    "warmup_runs": args.warmup,
                    "timed_runs": args.repetitions,
                    "peak_allocated_vram_mib": float(peak),
                    "incremental_peak_vram_mib": float(peak - baseline),
                    "backbone_lr": metrics["backbone_lr"],
                    "step1_best_epoch": metrics["best_epoch"],
                    "step1_test_accuracy": metrics["test"]["accuracy"],
                    "step1_test_f1_macro": metrics["test"]["f1_macro"],
                    "cv_accuracy_mean": float(cv.loc[backbone, "test_accuracy_mean"]),
                    "cv_accuracy_std": float(cv.loc[backbone, "test_accuracy_std"]),
                    "cv_f1_macro_mean": float(cv.loc[backbone, "test_f1_macro_mean"]),
                    "cv_f1_macro_std": float(cv.loc[backbone, "test_f1_macro_std"]),
                    "checkpoint_path": str(checkpoint.relative_to(ROOT)) if weights == "checkpoint" else None,
                }
            )
            sample_rows.extend(
                {"backbone": backbone, "weights": weights, "repetition": i + 1, "latency_ms": v}
                for i, v in enumerate(samples.tolist())
            )
            del model
            torch.cuda.empty_cache()

    results = pd.DataFrame(rows)
    results.to_csv(out / "efficiency_17class_rtx4090.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(out / "latency_samples.csv", index=False)
    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    metadata = {
        "created_utc": stamp,
        "scope": "Task B, 17 classes, step-1 checkpoints (backbone LR 1e-4, seed 42, Final-B split)",
        "protocol": {
            "input": [1, 3, 224, 224],
            "precision": "FP32",
            "warmup_runs": args.warmup,
            "timed_runs": args.repetitions,
            "latency_timer": "torch.cuda.Event; one synchronization per sample (phase2_benchmark.measure_latency)",
            "vram": "torch.cuda.max_memory_allocated after warmup",
            "macs": "THOP multiply-accumulate count",
            "torch_cpu_threads": args.threads,
            "control": "each architecture also timed with random weights (torch.manual_seed(0))",
        },
        "environment": environment,
        "woodbench_git_commit": commit,
        "note": "RTX 4090 numbers are not comparable with the RTX 3060 table in the paper; compare models within one GPU only.",
    }
    (out / "efficiency_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    show = results.pivot(index="display_name", columns="weights",
                         values=["latency_median_ms", "peak_allocated_vram_mib"]).round(2)
    print(show.to_string())
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
