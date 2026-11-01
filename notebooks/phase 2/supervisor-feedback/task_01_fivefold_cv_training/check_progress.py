#!/usr/bin/env python3
"""Monitor matching completion markers and live process identities."""

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import time

from run_artifacts import ROOT, completed_fold, digest, process_token

OUTPUT_ROOT = ROOT / "data/experiments/Task-B-origin-cv-v2"
MODELS = [
    "convnext_tiny",
    "mobilenetv3_large",
    "resnet50",
    "efficientnet_b3",
    "swin_tiny",
]
STATUSES = ("completed", "in_progress", "interrupted", "error", "not_started")


def get_model_progress(model_dir: Path, n_folds: int = 5):
    identity_path = model_dir / "run_config.json"
    identity, config_error = None, None
    try:
        if identity_path.exists():
            identity = json.loads(identity_path.read_text())
            if digest(identity["config"]) != identity["fingerprint"]:
                raise ValueError("Changed config fingerprint")
            n_folds = int(identity["config"]["n_folds"])
            if n_folds < 2:
                raise ValueError("Invalid fold count")
    except (ValueError, KeyError, OSError, TypeError) as error:
        config_error, n_folds = str(error), 5

    epochs = identity["config"].get("epochs") if identity and not config_error else None
    folds = []
    for fold in range(1, n_folds + 1):
        folder = model_dir / f"fold_{fold}"
        item = {
            "fold": fold,
            "status": "not_started",
            "accuracy": None,
            "epoch": None,
            "epochs": epochs,
            "best_epoch": None,
        }
        try:
            if config_error:
                raise ValueError(config_error)
            if (folder / "metrics.json").exists():
                if identity:
                    metrics = completed_fold(
                        folder, identity["fingerprint"], verify_checksums=False
                    )
                else:
                    metrics = json.loads((folder / "metrics.json").read_text())
                    if metrics.get("status") != "completed":
                        raise ValueError("Legacy completion marker")
                item.update(
                    status="completed",
                    accuracy=metrics["accuracy"],
                    epoch=metrics.get("epoch"),
                    best_epoch=metrics.get("best_epoch"),
                )
            elif (folder / "run.json").exists():
                run = json.loads((folder / "run.json").read_text())
                if identity and run.get("fingerprint") != identity["fingerprint"]:
                    raise ValueError("Unfinished fold config mismatch")
                token = run.get("process_token")
                alive = token is not None and process_token(run["pid"]) == token
                item.update(
                    status=(
                        "in_progress"
                        if run["status"] == "running" and alive
                        else "interrupted"
                    ),
                    epoch=run.get("epoch"),
                )
            elif any(
                (folder / name).exists()
                for name in ("last.pth", "final.pth", "best.pth")
            ):
                item["status"] = "interrupted"
        except (ValueError, KeyError, OSError, TypeError) as error:
            item.update(status="error", error=str(error))
        folds.append(item)

    counts = Counter(item["status"] for item in folds)
    accuracies = [item["accuracy"] for item in folds if item["accuracy"] is not None]
    return {
        "model": model_dir.name,
        "folds": folds,
        "accuracies": accuracies,
        **{status: counts[status] for status in STATUSES},
        "mean_accuracy": statistics.mean(accuracies) if accuracies else None,
        "std_accuracy": statistics.stdev(accuracies) if len(accuracies) > 1 else None,
    }


def print_progress(model_results, output_root, verbose=False):
    print(f"ORIGIN CV PROGRESS\nOutput: {output_root}")
    for result in model_results:
        counts = ", ".join(f"{status}={result[status]}" for status in STATUSES)
        print(f"{result['model']}: {counts}")
        if result["mean_accuracy"] is not None:
            print(
                f"  Held-out accuracy mean={result['mean_accuracy']:.4f}; fold std={result['std_accuracy']}"
            )
        if verbose:
            for fold in result["folds"]:
                print(
                    f"  Fold {fold['fold']}: {fold['status']}, "
                    f"epoch={fold['epoch'] or 0}/{fold['epochs'] or '?'}, "
                    f"accuracy={fold['accuracy']}"
                    + (f", best_epoch={fold['best_epoch']}" if fold["best_epoch"] else "")
                )
                if fold.get("error"):
                    print(f"    {fold['error']}")
    print(
        f"SUMMARY: {sum(r['completed'] for r in model_results)}/{sum(len(r['folds']) for r in model_results)} folds completed"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    models = [args.model] if args.model else MODELS
    try:
        while True:
            results = [get_model_progress(args.output / model) for model in models]
            print_progress(results, args.output, args.verbose)
            if not args.watch:
                return
            time.sleep(30)
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
