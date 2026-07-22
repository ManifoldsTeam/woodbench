#!/usr/bin/env python3
"""Launch multiple Hydra training runs (sequential or parallel).

All hyperparameters live in configs/; this script only forwards CLI overrides.

Examples:
    python scripts/run_experiments.py --experiment taskB --backbone convnext_tiny --seeds 42 123
    python scripts/run_experiments.py --experiment taskB --all_models --seeds 42 --max_workers 2
    python scripts/run_experiments.py --ablation 90rotation --backbone convnext_tiny --seeds 42
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent.parent
TRAIN_ENTRY = ROOT / "train.py"

ABLATION_TO_EXPERIMENT = {
    "90rotation": "taskB_90rotation",
    "label_smooth_0.0": "taskB_label_smoothing_0.0",
    "label_smooth_0.2": "taskB_label_smoothing_0.2",
    "resolution_384": "taskB_resolution_384",
}


@dataclass(frozen=True)
class RunSpec:
    experiment: str
    backbone: str
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch launcher for train.py (Hydra)")
    parser.add_argument(
        "--experiment",
        type=str,
        default="taskB",
        help="Hydra experiment config name under configs/experiment/",
    )
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--all_models", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument(
        "--ablation",
        type=str,
        choices=list(ABLATION_TO_EXPERIMENT.keys()),
        default=None,
    )
    parser.add_argument("--max_workers", type=int, default=2)
    parser.add_argument(
        "--output_root",
        type=str,
        default="experiments",
        help="Maps to Hydra output.root",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--disable_wandb", action="store_true")
    return parser.parse_args()


def resolve_backbones(args: argparse.Namespace) -> list[str]:
    from src.models import BACKBONE_NAMES

    if args.all_models:
        return list(BACKBONE_NAMES.keys())
    if args.backbone:
        return [args.backbone]
    raise SystemExit("Specify --backbone or --all_models")


def build_hydra_overrides(spec: RunSpec, args: argparse.Namespace) -> list[str]:
    overrides = [
        f"experiment={spec.experiment}",
        f"model.backbone={spec.backbone}",
        f"train.seed={spec.seed}",
        f"output.root={args.output_root}",
    ]
    if args.epochs is not None:
        overrides.append(f"train.epochs={args.epochs}")
    if args.batch_size is not None:
        overrides.append(f"data.dataloader.batch_size={args.batch_size}")
    if args.disable_wandb:
        overrides.append("train.wandb.enabled=false")
    return overrides


def run_one(spec: RunSpec, args: argparse.Namespace, log_file: Path) -> dict:
    cmd = [sys.executable, str(TRAIN_ENTRY), *build_hydra_overrides(spec, args)]
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with open(log_file, "w") as log:
        completed = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)

    return {
        "spec": spec,
        "cmd": " ".join(cmd),
        "returncode": completed.returncode,
        "log_file": str(log_file),
    }


def main() -> None:
    args = parse_args()
    experiment = (
        ABLATION_TO_EXPERIMENT[args.ablation] if args.ablation else args.experiment
    )
    backbones = resolve_backbones(args)

    specs = [
        RunSpec(experiment=experiment, backbone=backbone, seed=seed)
        for backbone in backbones
        for seed in args.seeds
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ROOT / args.output_root / "logs" / f"batch_{timestamp}"

    print(f"Runs: {len(specs)} | workers: {args.max_workers} | experiment: {experiment}")

    if args.max_workers <= 1:
        results = []
        for index, spec in enumerate(specs, start=1):
            log_file = log_dir / f"{spec.backbone}_seed{spec.seed}.log"
            print(f"[{index}/{len(specs)}] {spec.backbone} seed={spec.seed}")
            results.append(run_one(spec, args, log_file))
    else:
        tasks = [
            (spec, args, log_dir / f"{spec.backbone}_seed{spec.seed}.log")
            for spec in specs
        ]
        with mp.Pool(processes=args.max_workers) as pool:
            results = pool.starmap(run_one, tasks)

    success = sum(1 for r in results if r["returncode"] == 0)
    summary_path = log_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "experiment": experiment,
                "total": len(specs),
                "successful": success,
                "failed": len(specs) - success,
                "results": [
                    {
                        "backbone": r["spec"].backbone,
                        "seed": r["spec"].seed,
                        "returncode": r["returncode"],
                        "log_file": r["log_file"],
                    }
                    for r in results
                ],
            },
            f,
            indent=2,
        )

    print(f"Done: {success}/{len(specs)} OK | summary: {summary_path}")


if __name__ == "__main__":
    main()
