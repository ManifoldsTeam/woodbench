#!/usr/bin/env python3
"""Tables for supervisor requests 6-7 (14 species), after run_clean14.sh.

Reads data/experiments/Task-B-clean14-{cv,cv-grayscale,cv-px16,final}/ and
writes artifacts/phase_2/supervisor-feedback/task_15_clean14_runs/results/:

  summary.md                     all tables below, as text
  cv_summary.csv                 per model: held-out acc / macro-F1, mean +- SD over 5 folds
  cv_folds.csv                   per fold: best epoch, epochs trained, acc, F1
  cv_vs_17species.csv            same folds, same 14 species: CV v2 (17-species model,
                                 01-11 rule) vs this CV
  degraded_summary.csv           ConvNeXt rgb / grayscale / px16, per fold and mean
  final_b_runs.csv               per Final-B run: best epoch, val/test acc, macro-F1
  final_b_summary.csv            per model: mean +- SD over seeds
  final_b_per_species_f1.csv     test F1 per species, mean over seeds
  checkpoints.sha256             every best.pth (checkpoints stay out of git)
  runs/                          copies of every run's json/csv files (no .pth)

CV reports (registry, OOF, per-species) also go to <cv root>/reports/ through
task_01's cv_reports.export_reports, as for CV v2. SD is the sample SD.
"""

import argparse
import csv
import json
from pathlib import Path
import shutil
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE.parent / "task_01_fivefold_cv_training"))
from cv_reports import export_reports  # noqa: E402
from origin_cv import SPECIES_SETS  # noqa: E402
from run_artifacts import file_sha256  # noqa: E402

CV_ROOTS = {"rgb": "Task-B-clean14-cv", "grayscale": "Task-B-clean14-cv-grayscale",
            "px16": "Task-B-clean14-cv-px16"}
FINAL_ROOT = "Task-B-clean14-final"
CV_V2 = ROOT / "artifacts/phase_2/supervisor-feedback/task_01_fivefold_cv_training/results-v2"
MODEL_ORDER = ["convnext_tiny", "swin_tiny", "efficientnet_b3", "resnet50", "mobilenetv3_large"]


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def relative(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def mean_sd(values):
    values = [float(v) for v in values]
    sd = statistics.stdev(values) if len(values) > 1 else float("nan")
    return statistics.mean(values), sd


def pct(values):
    mean, sd = mean_sd(values)
    return f"{100 * mean:.2f} ± {100 * sd:.2f}"


def order(name):
    return MODEL_ORDER.index(name) if name in MODEL_ORDER else len(MODEL_ORDER)


def cv_folds(root):
    rows = []
    for path in sorted(root.glob("*/fold_*/metrics.json")):
        m = json.loads(path.read_text())
        rows.append({
            "backbone": m["backbone"], "input_variant": m["config"]["input_variant"],
            "fold": m["fold"], "best_epoch": m["best_epoch"], "epochs_trained": m["epochs_trained"],
            "best_val_loss": m["best_val_loss"], "best_val_accuracy": m["best_val_accuracy"],
            "n_test": m["n_test"], "accuracy": m["accuracy"], "f1_macro": m["f1_macro"],
            "run_id": m["run_id"],
        })
    return sorted(rows, key=lambda r: (order(r["backbone"]), r["fold"]))


def cv_v2_on_clean14(backbone, fold):
    """01-11 CV v2 (17-species model) accuracy on the images of the 14 species."""
    excluded = set(SPECIES_SETS["clean14"])
    path = CV_V2 / backbone / f"fold_{fold}" / "predictions.csv"
    if not path.exists():
        return None
    rows = [r for r in read_csv(path) if r["true_species"] not in excluded]
    return sum(r["true_species"] == r["predicted_species"] for r in rows) / len(rows), len(rows)


def copy_light(source, target):
    """Copy every .json/.csv (run files and CV reports); checkpoints stay out."""
    for path in source.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".csv"}:
            destination = target / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exp_root", type=Path, default=ROOT / "data/experiments")
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "artifacts/phase_2/supervisor-feedback/task_15_clean14_runs/results",
    )
    args = parser.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    md = ["# 14-species runs (supervisor requests 6-7, 06-11-2026)", ""]
    checkpoints = []

    folds = {}
    for variant, name in CV_ROOTS.items():
        root = args.exp_root / name
        if not root.exists():
            md += [f"`{name}`: not found, skipped.", ""]
            continue
        registry = export_reports(root)  # raises on incomplete CV
        md += [f"CV report `{name}`: `{Path(registry).parent.name}`", ""]
        folds[variant] = cv_folds(root)
        copy_light(root, out / "runs" / name)
        checkpoints += sorted(root.glob("*/fold_*/best.pth"))

    if "rgb" in folds:
        rows, compare = [], []
        for backbone in sorted({r["backbone"] for r in folds["rgb"]}, key=order):
            fr = [r for r in folds["rgb"] if r["backbone"] == backbone]
            rows.append({
                "backbone": backbone, "n_folds": len(fr),
                "accuracy_mean": mean_sd(r["accuracy"] for r in fr)[0],
                "accuracy_sd": mean_sd(r["accuracy"] for r in fr)[1],
                "f1_macro_mean": mean_sd(r["f1_macro"] for r in fr)[0],
                "f1_macro_sd": mean_sd(r["f1_macro"] for r in fr)[1],
                "oof_accuracy": sum(r["accuracy"] * r["n_test"] for r in fr) / sum(r["n_test"] for r in fr),
                "best_epoch_mean": statistics.mean(r["best_epoch"] for r in fr),
                "epochs_trained_mean": statistics.mean(r["epochs_trained"] for r in fr),
            })
            for r in fr:
                old = cv_v2_on_clean14(backbone, r["fold"])
                if old:
                    compare.append({"backbone": backbone, "fold": r["fold"],
                                    "cv_v2_17species_on_14": old[0], "n_images": old[1],
                                    "clean14": r["accuracy"], "n_clean14": r["n_test"],
                                    "difference": r["accuracy"] - old[0]})
        write_csv(out / "cv_folds.csv", folds["rgb"])
        write_csv(out / "cv_summary.csv", rows)
        md += ["## CV, 5 models × 5 folds (held-out fold, best.pth)", "",
               "| Model | Acc (%) | Macro-F1 (%) | OOF acc (%) | Best epoch | Epochs |",
               "|---|---:|---:|---:|---:|---:|"]
        for backbone in [r["backbone"] for r in rows]:
            fr = [r for r in folds["rgb"] if r["backbone"] == backbone]
            row = next(r for r in rows if r["backbone"] == backbone)
            md.append(f"| {backbone} | {pct(r['accuracy'] for r in fr)} | {pct(r['f1_macro'] for r in fr)} | "
                      f"{100 * row['oof_accuracy']:.2f} | {row['best_epoch_mean']:.1f} | "
                      f"{row['epochs_trained_mean']:.1f} |")
        md.append("")
        if compare:
            write_csv(out / "cv_vs_17species.csv", compare)
            md += ["## Same folds, same 14 species: 17-species model (CV v2, 01-11 rule) vs 14-species model",
                   "", "CV v2 accuracy counts a prediction of one of the 3 removed species as wrong.", "",
                   "| Model | CV v2 on 14 species (%) | 14-species CV (%) | Mean difference (pts) |",
                   "|---|---:|---:|---:|"]
            for backbone in [r["backbone"] for r in rows]:
                c = [r for r in compare if r["backbone"] == backbone]
                if c:
                    md.append(f"| {backbone} | {pct(r['cv_v2_17species_on_14'] for r in c)} | "
                              f"{pct(r['clean14'] for r in c)} | "
                              f"{100 * statistics.mean(r['difference'] for r in c):+.2f} |")
            md.append("")

    degraded = [r for v in ("rgb", "grayscale", "px16") for r in folds.get(v, [])
                if r["backbone"] == "convnext_tiny"]
    if any(r["input_variant"] != "rgb" for r in degraded):
        write_csv(out / "degraded_summary.csv", degraded)
        md += ["## ConvNeXt-Tiny, input degradation (same 5 folds)", "",
               "| Input | Acc per fold (%) | Acc (%) | Macro-F1 (%) | Best epoch |", "|---|---|---:|---:|---:|"]
        for variant in ("rgb", "grayscale", "px16"):
            fr = sorted((r for r in degraded if r["input_variant"] == variant), key=lambda r: r["fold"])
            if fr:
                md.append(f"| {variant} | {', '.join(f'{100 * r['accuracy']:.2f}' for r in fr)} | "
                          f"{pct(r['accuracy'] for r in fr)} | {pct(r['f1_macro'] for r in fr)} | "
                          f"{statistics.mean(r['best_epoch'] for r in fr):.1f} |")
        md.append("")

    final_root = args.exp_root / FINAL_ROOT
    runs = []
    for path in sorted(final_root.glob("seed_*/*/metrics.json")):
        m = json.loads(path.read_text())
        for name, digest in m["artifact_sha256"].items():
            if file_sha256(path.parent / name) != digest:
                raise ValueError(f"Changed artifact: {path.parent / name}")
        runs.append((m, path.parent))
    if runs:
        copy_light(final_root, out / "runs" / FINAL_ROOT)
        checkpoints += [d / "best.pth" for _, d in runs]
        rows = [{
            "backbone": m["backbone"], "seed": m["seed"], "best_epoch": m["best_epoch"],
            "epochs_trained": m["epochs_trained"], "val_loss": m["val"]["loss"],
            "val_accuracy": m["val"]["accuracy"], "val_f1_macro": m["val"]["f1_macro"],
            "test_accuracy": m["test"]["accuracy"], "test_f1_macro": m["test"]["f1_macro"],
            "test_errors": round(m["test"]["n"] * (1 - m["test"]["accuracy"])), "n_test": m["test"]["n"],
            "run_id": m["run_id"],
        } for m, _ in runs]
        rows.sort(key=lambda r: (order(r["backbone"]), r["seed"]))
        write_csv(out / "final_b_runs.csv", rows)
        summary, species = [], []
        md += ["## Final-B, 5 models × 3 seeds (test 432 images, best.pth)", "",
               "| Model | Seeds | Test acc (%) | Test macro-F1 (%) | Val acc (%) | Best epoch per seed |",
               "|---|---:|---:|---:|---:|---|"]
        for backbone in sorted({r["backbone"] for r in rows}, key=order):
            br = [r for r in rows if r["backbone"] == backbone]
            summary.append({
                "backbone": backbone, "n_seeds": len(br),
                **{f"{k}_{s}": v for k in ("test_accuracy", "test_f1_macro", "val_accuracy")
                   for s, v in zip(("mean", "sd"), mean_sd(r[k] for r in br))},
            })
            md.append(f"| {backbone} | {len(br)} | {pct(r['test_accuracy'] for r in br)} | "
                      f"{pct(r['test_f1_macro'] for r in br)} | {pct(r['val_accuracy'] for r in br)} | "
                      f"{', '.join(str(r['best_epoch']) for r in br)} |")
            per_run = [m["test"]["f1_per_class"] for m, _ in runs if m["backbone"] == backbone]
            for name in per_run[0]:
                species.append({"backbone": backbone, "species": name,
                                "test_f1_mean": statistics.mean(p[name] for p in per_run),
                                "n_seeds": len(per_run)})
        write_csv(out / "final_b_summary.csv", summary)
        write_csv(out / "final_b_per_species_f1.csv", species)
        md.append("")

    if checkpoints:
        (out / "checkpoints.sha256").write_text("".join(
            f"{file_sha256(p)}  {relative(p)}\n" for p in checkpoints))
        md += [f"{len(checkpoints)} checkpoints listed in `checkpoints.sha256`.", ""]
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
