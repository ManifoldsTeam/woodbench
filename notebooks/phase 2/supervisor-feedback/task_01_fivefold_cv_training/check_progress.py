#!/usr/bin/env python3
"""
5-Fold CV Progress Monitor

Check progress of all running/finished k-fold experiments.

Usage:
    python check_progress.py
    python check_progress.py --watch
    python check_progress.py --model convnext_tiny
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
import sys

ROOT = Path(__file__).parent.parent.parent.parent.resolve()
OUTPUT_ROOT = ROOT / "data" / "experiments" / "Task-B-origin-cv"

MODELS = ["convnext_tiny", "mobilenetv3_large", "resnet50", "efficientnet_b3", "swin_tiny"]


def get_model_progress(model_dir: Path, n_folds: int = 5):
    """Get progress for one model."""
    results = {
        "model": model_dir.name,
        "folds": [],
        "completed": 0,
        "in_progress": 0,
        "not_started": 0,
        "accuracies": [],
    }

    for fold in range(1, n_folds + 1):
        fold_dir = model_dir / f"fold_{fold}"
        metrics_path = fold_dir / "metrics.json"
        best_path = fold_dir / "best.pth"

        fold_result = {
            "fold": fold,
            "status": "not_started",
            "accuracy": None,
            "best_epoch": None,
        }

        if metrics_path.exists():
            try:
                with open(metrics_path) as f:
                    metrics = json.load(f)
                fold_result["status"] = "completed"
                fold_result["accuracy"] = metrics.get("accuracy", metrics.get("test_accuracy"))
                fold_result["best_epoch"] = metrics.get("best_epoch")
                results["completed"] += 1
                if fold_result["accuracy"]:
                    results["accuracies"].append(fold_result["accuracy"])
            except:
                fold_result["status"] = "error"
        elif best_path.exists():
            fold_result["status"] = "in_progress"
            results["in_progress"] += 1
        else:
            results["not_started"] += 1

        results["folds"].append(fold_result)

    if results["accuracies"]:
        import numpy as np
        results["mean_accuracy"] = np.mean(results["accuracies"])
        results["std_accuracy"] = np.std(results["accuracies"])
    else:
        results["mean_accuracy"] = None
        results["std_accuracy"] = None

    return results


def print_progress(model_results, verbose=False):
    """Print progress for all models."""
    print("\n" + "=" * 80)
    print("5-FOLD CV PROGRESS - Origin Key Split")
    print("=" * 80)
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)

    total_completed = 0
    total_folds = 0

    for result in model_results:
        model = result["model"]
        completed = result["completed"]
        in_progress = result["in_progress"]
        not_started = result["not_started"]
        total = completed + in_progress + not_started
        total_folds += total

        # Status indicator
        if completed == 5:
            status = "✅ DONE"
        elif completed > 0 or in_progress > 0:
            status = "🔄 IN PROGRESS"
        else:
            status = "⏳ NOT STARTED"

        print(f"\n{model.upper():30s} {status}")
        print(f"  Progress: [{'●' * completed}{'◐' * in_progress}{'○' * not_started}] {completed}/5 folds")

        if result["mean_accuracy"] is not None:
            mean_acc = result["mean_accuracy"]
            std_acc = result["std_accuracy"]
            print(f"  Accuracy: {mean_acc:.4f} ± {std_acc:.4f} ({mean_acc*100:.2f}%)")
        elif completed > 0:
            print(f"  Completed folds: {completed}")

        if verbose:
            for fold in result["folds"]:
                status_icon = {
                    "completed": "✅",
                    "in_progress": "🔄",
                    "not_started": "○",
                    "error": "❌"
                }.get(fold["status"], "?")

                acc_str = f"{fold['accuracy']:.4f}" if fold["accuracy"] else "---"
                epoch_str = f"ep{fold['best_epoch']}" if fold["best_epoch"] else ""

                print(f"    Fold {fold['fold']}: {status_icon} {acc_str} {epoch_str}")

    # Summary
    total_completed = sum(r["completed"] for r in model_results)
    print("\n" + "=" * 80)
    print(f"SUMMARY: {total_completed}/{len(model_results) * 5} folds completed across {len(model_results)} models")

    # Estimated time remaining
    if total_completed < len(model_results) * 5:
        avg_time_per_fold = 30  # minutes (rough estimate)
        remaining = (len(model_results) * 5) - total_completed
        print(f"Estimated remaining: ~{remaining * avg_time_per_fold / 60:.1f} hours (if running sequentially)")
    print("=" * 80)


def check_wandb_status():
    """Check WandB for running experiments."""
    try:
        import wandb
        api = wandb.Api()

        print("\n--- WandB Running Experiments ---")
        runs = api.runs("your-username/woodbench-origin-cv", filters={"state": "running"})

        if runs:
            for run in runs:
                print(f"  {run.name}: {run.state}")
        else:
            print("  No running experiments found")
    except Exception as e:
        print(f"  (WandB not available or not logged in)")


def main():
    parser = argparse.ArgumentParser(description='Check 5-fold CV progress')
    parser.add_argument('--watch', action='store_true', help='Watch mode (refresh every 30s)')
    parser.add_argument('--model', type=str, help='Check specific model only')
    parser.add_argument('--output', type=str, help='Output root path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    global OUTPUT_ROOT
    if args.output:
        OUTPUT_ROOT = Path(args.output)

    models_to_check = [args.model] if args.model else MODELS

    def do_check():
        model_results = []
        for model in models_to_check:
            model_dir = OUTPUT_ROOT / model
            if model_dir.exists():
                result = get_model_progress(model_dir)
                model_results.append(result)
            else:
                model_results.append({
                    "model": model,
                    "completed": 0,
                    "folds": [],
                    "mean_accuracy": None,
                })

        print_progress(model_results, verbose=args.verbose)

        if os.getenv("WANDB_API_KEY"):
            check_wandb_status()

    if args.watch:
        import time
        print("Watching progress... (Ctrl+C to stop)")
        try:
            while True:
                os.system('clear' if os.name == 'posix' else 'cls')
                do_check()
                time.sleep(30)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        do_check()


if __name__ == "__main__":
    main()
