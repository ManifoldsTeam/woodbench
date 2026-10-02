"""Postprocessing used by evaluate_and_register_cv.ipynb, without retraining."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from run_artifacts import (
    PROTOCOL,
    completed_fold,
    digest,
    relative_path,
    save_csv,
    save_json,
)


def export_reports(output_root):
    output_root = Path(output_root)
    registry_rows, per_class, predictions, summaries = [], [], [], []
    for model_dir in sorted(output_root.iterdir()):
        config_path = model_dir / "run_config.json"
        if not config_path.is_file():
            continue
        identity = json.loads(config_path.read_text())
        config, fingerprint = identity["config"], identity["fingerprint"]
        if config["protocol"] != PROTOCOL or digest(config) != fingerprint:
            raise ValueError(f"Unsupported or changed run config: {config_path}")
        model_rows, model_predictions = [], []
        for fold in range(1, config["n_folds"] + 1):
            folder = model_dir / f"fold_{fold}"
            run = completed_fold(folder, fingerprint)
            if run is None:
                raise ValueError(
                    f"Incomplete CV: {folder}; finish training before report export"
                )
            rows = list(csv.DictReader((folder / "predictions.csv").open()))
            if len(rows) != run["n_test"] or any(
                r["run_id"] != run["run_id"] for r in rows
            ):
                raise ValueError(f"Prediction count/run identity mismatch: {folder}")
            classes = run["class_to_idx"]
            labels = list(range(len(classes)))
            for row in rows:
                if classes.get(row["true_species"]) != int(
                    row["true_label"]
                ) or classes.get(row["predicted_species"]) != int(
                    row["predicted_label"]
                ):
                    raise ValueError(f"Prediction class mapping mismatch: {folder}")
            y = [int(row["true_label"]) for row in rows]
            p = [int(row["predicted_label"]) for row in rows]
            precision, recall, f1, support = precision_recall_fscore_support(
                y, p, labels=labels, zero_division=0
            )
            metrics = {
                "test_accuracy": accuracy_score(y, p),
                "test_precision_macro": float(np.mean(precision)),
                "test_recall_macro": float(np.mean(recall)),
                "test_f1_macro": float(np.mean(f1)),
            }
            row = {
                key: run[key]
                for key in (
                    "run_id",
                    "run_date",
                    "notebook",
                    "git_commit",
                    "split_version",
                    "num_classes",
                    "n_train",
                    "n_val",
                    "n_test",
                    "checkpoint_path",
                    "prediction_path",
                    "backbone",
                    "fold",
                    "fingerprint",
                )
            }
            row.update(
                seed=config["seed"],
                lr=json.dumps(
                    {"backbone": config["backbone_lr"], "head": config["head_lr"]}
                ),
                epochs=config["epochs"],
                batch=config["batch_size"],
                image_size=config["image_size"],
                label_smoothing=config["label_smoothing"],
                augmentation=config["augmentation"],
                protocol=PROTOCOL,
                class_to_idx=json.dumps(classes, sort_keys=True),
                code_sha256=config["code_sha256"],
                **metrics,
            )
            registry_rows.append(row)
            model_rows.append(row)
            for name, index in sorted(classes.items(), key=lambda pair: pair[1]):
                per_class.append(
                    {
                        "run_id": run["run_id"],
                        "backbone": run["backbone"],
                        "fold": fold,
                        "species": name,
                        "precision": precision[index],
                        "recall": recall[index],
                        "f1": f1[index],
                        "support": int(support[index]),
                    }
                )
            for prediction in rows:
                prediction["backbone"] = run["backbone"]
            model_predictions.extend(rows)
        expected = model_rows[0]["n_train"] + model_rows[0]["n_test"]
        if (
            len(model_predictions) != expected
            or len({r["path"] for r in model_predictions}) != expected
        ):
            raise ValueError(f"OOF does not cover every image once: {model_dir}")
        for key in ("origin_key", "md5"):
            folds = {}
            for row in model_predictions:
                folds.setdefault(row[key], set()).add(row["fold"])
            if any(len(values) != 1 for values in folds.values()):
                raise ValueError(f"OOF {key} crosses folds: {model_dir}")
        predictions.extend(model_predictions)
        summary = {
            "backbone": model_rows[0]["backbone"],
            "n_folds": len(model_rows),
            "split_version": config["split_version"],
            "source_run_ids": json.dumps([r["run_id"] for r in model_rows]),
        }
        for metric in (
            "test_accuracy",
            "test_precision_macro",
            "test_recall_macro",
            "test_f1_macro",
        ):
            values = [r[metric] for r in model_rows]
            summary[metric + "_mean"] = float(np.mean(values))
            summary[metric + "_std"] = float(np.std(values, ddof=1))
        summary["oof_accuracy"] = accuracy_score(
            [r["true_label"] for r in model_predictions],
            [r["predicted_label"] for r in model_predictions],
        )
        summaries.append(summary)
    if not registry_rows:
        raise ValueError(f"No completed CV runs in {output_root}")
    if len({r["run_id"] for r in registry_rows}) != len(registry_rows):
        raise ValueError("Duplicate training run IDs")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = output_root / "reports" / f"cv-{stamp}-{uuid.uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    registry = directory / f"origin-cv-{stamp}-registry.csv"
    save_csv(registry, registry_rows)
    save_csv(directory / "per_class.csv", per_class)
    save_csv(directory / "summary.csv", summaries)
    save_csv(directory / "oof_predictions.csv", predictions)
    save_json(
        directory / "metadata.json",
        {
            "registry": relative_path(registry),
            "created_at": stamp,
            "protocol": PROTOCOL,
            "std": "sample standard deviation across folds; not CI",
            "source_run_ids": [r["run_id"] for r in registry_rows],
            "validation": "No inner validation; fixed epochs, outer held-out evaluated once",
            "comparison": "Fixed-split baseline comparison requires explicit baseline_run_id and registry",
        },
    )
    return registry
