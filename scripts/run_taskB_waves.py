#!/usr/bin/env python3
"""Run Task B training in 3 waves across 2 vast.ai instances.

Wave 1 (parallel): ConvNeXt + Swin
Wave 2 (parallel): MobileNet + ResNet
Wave 3 (sequential): EfficientNet
Final: Results Driver

Usage:
    # Wave 1
    python scripts/run_taskB_waves.py --wave 1 --kernel_name woodbench

    # All waves sequentially
    python scripts/run_taskB_waves.py --all --kernel_name woodbench

Requirements:
    pip install papermill jupyter nbconvert
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).parent.parent / "notebooks" / "phase 1" / "task-B"
OUT_DIR = NOTEBOOK_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

NOTEBOOKS = {
    "convnext": "TaskB_convnext_tiny.ipynb",
    "swin": "TaskB_swin_tiny.ipynb",
    "mobilenet": "TaskB_mobilenetv3_large.ipynb",
    "resnet": "TaskB_resnet50.ipynb",
    "efficientnet": "TaskB_efficientnet_b3.ipynb",
    "driver": "TaskB_Results_Driver.ipynb",
}

WAVES = [
    {
        "name": "Wave 1 — ConvNeXt + Swin (parallel)",
        "jobs": [("convnext", "swin")],
    },
    {
        "name": "Wave 2 — MobileNet + ResNet (parallel)",
        "jobs": [("mobilenet", "resnet")],
    },
    {
        "name": "Wave 3 — EfficientNet (sequential)",
        "jobs": [("efficientnet",)],
    },
]


def run_notebook(name: str, kernel: str, timeout: int | None = None) -> bool:
    """Run one notebook via papermill. Returns True on success."""
    nb_path = NOTEBOOK_DIR / NOTEBOOKS[name]
    out_path = OUT_DIR / f"{name}_output.ipynb"

    cmd = [
        sys.executable, "-m", "papermill",
        str(nb_path),
        str(out_path),
        "-k", kernel,
    ]
    if timeout:
        cmd.extend(["--timeout", str(timeout)])

    print(f"\n[START] {NOTEBOOKS[name]} → {out_path.name}")
    result = subprocess.run(cmd, capture_output=False)
    ok = result.returncode == 0
    status = "DONE" if ok else "FAILED"
    print(f"\n[{status}] {NOTEBOOKS[name]}")
    return ok


def run_parallel_pair(
    job_a: str, job_b: str, kernel: str, timeout: int | None = None
) -> tuple[bool, bool]:
    """Run 2 notebooks in parallel threads."""
    print(f"\n{'='*65}")
    print(f"PARALLEL: {job_a}  ||  {job_b}")
    print(f"{'='*65}")

    results = {}

    def worker(name: str):
        results[name] = run_notebook(name, kernel, timeout)

    t_a = threading.Thread(target=worker, args=(job_a,))
    t_b = threading.Thread(target=worker, args=(job_b,))

    t_a.start()
    t_b.start()

    t_a.join()
    t_b.join()

    return results.get(job_a, False), results.get(job_b, False)


def run_wave(wave: dict, kernel: str, timeout: int | None) -> list[bool]:
    """Run one wave: parallel pair(s) sequentially."""
    print(f"\n{'#'*65}")
    print(f"#  {wave['name']}")
    print(f"{'#'*65}")

    results = []
    for job in wave["jobs"]:
        if len(job) == 2:
            ra, rb = run_parallel_pair(job[0], job[1], kernel, timeout)
            results.extend([ra, rb])
        else:
            r = run_notebook(job[0], kernel, timeout)
            results.append(r)
        time.sleep(5)  # small delay between pairs

    return results


def run_driver(kernel: str, timeout: int | None = None) -> bool:
    print(f"\n{'#'*65}")
    print(f"#  Results Driver — wandb aggregation")
    print(f"{'#'*65}")
    return run_notebook("driver", kernel, timeout)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task B training waves")
    parser.add_argument("--wave", type=int, choices=[1, 2, 3], default=None,
                        help="Run specific wave only (1, 2, or 3)")
    parser.add_argument("--all", action="store_true",
                        help="Run all waves sequentially (1 → 2 → 3 → driver)")
    parser.add_argument("--driver", action="store_true",
                        help="Run results driver only")
    parser.add_argument("--kernel", dest="kernel", default="Python 3",
                        help="Jupyter kernel name (default: Python 3)")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Per-notebook timeout in seconds (default: no limit)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.driver:
        run_driver(args.kernel, args.timeout)
        return

    if args.wave:
        wave_idx = args.wave - 1
        results = run_wave(WAVES[wave_idx], args.kernel, args.timeout)
        ok = all(results)
        print(f"\n{'='*65}")
        print(f"Wave {args.wave} {'PASSED' if ok else 'FAILED'}")
        print(f"{'='*65}")
        return

    if args.all:
        print("="*65)
        print("Task B Training — All Waves")
        print("="*65)
        wave_results = []
        for i, wave in enumerate(WAVES):
            results = run_wave(wave, args.kernel, args.timeout)
            wave_results.extend(results)
            ok = all(results)
            print(f"\n[Wave {i+1}] {'PASSED' if ok else 'FAILED'} — "
                  f"{(i+1)/len(WAVES)*100:.0f}% complete")

        run_driver(args.kernel, args.timeout)

        all_ok = all(wave_results)
        print("\n" + "="*65)
        print(f"ALL WAVES {'COMPLETED' if all_ok else 'HAD FAILURES'}")
        print("="*65)
        return

    print("Usage: run_taskB_waves.py --all          # all waves + driver")
    print("       run_taskB_waves.py --wave 1      # wave 1 only")
    print("       run_taskB_waves.py --driver      # results driver only")


if __name__ == "__main__":
    main()
