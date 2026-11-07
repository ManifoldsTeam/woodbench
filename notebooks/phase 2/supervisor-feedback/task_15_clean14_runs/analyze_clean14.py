#!/usr/bin/env python3
"""Analysis of the 14-species runs (supervisor requests 6-7), after Vast.

Inputs (data/experiments/, from drive:wood/2026-11-07-clean14-runs/):
  Task-B-clean14-cv, -cv-grayscale, -cv-px16, -final, plus
  CV v2 predictions (artifacts/.../task_01_fivefold_cv_training/results-v2) and
  the August Final-B registry (docs/reports/experiments/phase_1b/final_b_new_recomputed/20260819T043733Z).

Writes artifacts/phase_2/supervisor-feedback/task_15_clean14_runs/analysis/:
  integrity.csv            metrics.json vs recomputed from predictions; code commit, data, versions
  convergence.csv          every run: epochs, best epoch, val acc at epochs 30/40/best, val-loss gain after epoch 40
  paired_tests.csv         exact McNemar tests on identical images (see `comparison` column)
  august_vs_clean14.csv    Final-B per model: August 17-species runs vs 14-species runs, same 432 test images
  oof_errors_by_origin.csv CV out-of-fold errors per origin, per model (14-species RGB)
  final_b_test_errors.csv  every misclassified Final-B test image, with the runs that miss it
  confusions.csv           most frequent true -> predicted pairs (CV OOF and Final-B test)
  degraded_per_species.csv ConvNeXt OOF F1 per species: rgb / grayscale / px16
  curves.png               val loss and val accuracy per epoch (CV mean over folds; Final-B mean over seeds)
"""

import csv
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import binomtest  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE.parent / "task_01_fivefold_cv_training"))
from origin_cv import PHASE1_MANIFEST, SPECIES_SETS, load_origin_samples  # noqa: E402
from run_artifacts import file_sha256  # noqa: E402

EXP = ROOT / "data/experiments"
OUT = ROOT / "artifacts/phase_2/supervisor-feedback/task_15_clean14_runs/analysis"
CV_V2 = ROOT / "artifacts/phase_2/supervisor-feedback/task_01_fivefold_cv_training/results-v2"
AUGUST = ROOT / "docs/reports/experiments/phase_1b/final_b_new_recomputed/20260819T043733Z/predictions.csv"
CACHE = ROOT / "data/task-B-group-split-224"
MODELS = ["convnext_tiny", "swin_tiny", "efficientnet_b3", "resnet50", "mobilenetv3_large"]
SEEDS = [42, 123, 456]
EXCLUDED = set(SPECIES_SETS["clean14"])


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(name, rows):
    with open(OUT / name, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def key(path):
    """split/species/file, independent of the data root (full-res or cache)."""
    return "/".join(Path(path).parts[-3:])


def mcnemar(a, b):
    """a, b: dict image -> correct (bool), same keys. Exact two-sided test."""
    only_a = sum(a[k] and not b[k] for k in a)
    only_b = sum(b[k] and not a[k] for k in a)
    p = binomtest(only_a, only_a + only_b).pvalue if only_a + only_b else 1.0
    return only_a, only_b, p


def cv_predictions(root, model):
    rows = []
    for fold in range(1, 6):
        rows += read_csv(root / model / f"fold_{fold}" / "predictions.csv")
    return rows


def final_predictions(model, seed, split="test"):
    return read_csv(EXP / "Task-B-clean14-final" / f"seed_{seed}" / model / f"predictions_{split}.csv")


def integrity():
    rows = []
    for path in sorted(EXP.glob("Task-B-clean14-cv*/*/fold_*/metrics.json")):
        m = json.loads(path.read_text())
        p = read_csv(path.parent / "predictions.csv")
        recomputed = sum(r["true_species"] == r["predicted_species"] for r in p) / len(p)
        rows.append({
            "run": path.parent.relative_to(EXP).as_posix(), "reported_accuracy": m["accuracy"],
            "recomputed_accuracy": recomputed, "n": len(p), "git_commit": m["git_commit"][:7],
            "data": "", "torch": "", "checkpoint_sha256_ok": file_sha256(path.parent / "best.pth")
            == m["artifact_sha256"]["best.pth"],
        })
    for path in sorted(EXP.glob("Task-B-clean14-final/seed_*/*/metrics.json")):
        m = json.loads(path.read_text())
        config = json.loads((path.parent / "run_config.json").read_text())
        p = read_csv(path.parent / "predictions_test.csv")
        rows.append({
            "run": path.parent.relative_to(EXP).as_posix(), "reported_accuracy": m["test"]["accuracy"],
            "recomputed_accuracy": sum(r["true_species"] == r["predicted_species"] for r in p) / len(p),
            "n": len(p), "git_commit": config["git_commit"][:7],
            "data": config["data_checksums_sha256"][:12], "torch": config["versions"]["torch"],
            "checkpoint_sha256_ok": file_sha256(path.parent / "best.pth") == m["artifact_sha256"]["best.pth"],
        })
    write_csv("integrity.csv", rows)
    bad = [r for r in rows if abs(r["reported_accuracy"] - r["recomputed_accuracy"]) > 1e-12
           or not r["checkpoint_sha256_ok"]]
    return len(rows), bad, sorted({r["git_commit"] for r in rows}), sorted({r["torch"] for r in rows} - {""}), \
        sorted({r["data"] for r in rows} - {""})


def convergence():
    rows = []
    paths = sorted(EXP.glob("Task-B-clean14-cv*/*/fold_*/history.json")) + sorted(
        EXP.glob("Task-B-clean14-final/seed_*/*/history.json"))
    for path in paths:
        h = json.loads(path.read_text())
        best = next(r for r in h if r["best"] and r["epoch"] == max(x["epoch"] for x in h if x["best"]))
        val_acc = {r["epoch"]: r["val_accuracy"] for r in h}
        val_loss = {r["epoch"]: r["val_loss"] for r in h}
        rows.append({
            "run": path.parent.relative_to(EXP).as_posix(),
            "epochs_trained": len(h), "best_epoch": best["epoch"], "reached_cap": len(h) == 50,
            "val_acc_epoch30": val_acc.get(30, ""), "val_acc_epoch40": val_acc.get(40, ""),
            "val_acc_best": best["val_accuracy"],
            "val_loss_gain_after_epoch40": val_loss[40] - best["val_loss"] if 40 in val_loss and best["epoch"] > 40 else 0.0,
            "lr_at_epoch45": next(iter(h[44]["lr"].values())) if len(h) >= 45 else "",
        })
    write_csv("convergence.csv", rows)
    return rows


def paired_tests():
    tests, cache = [], {}
    clean = {}
    for model in MODELS:
        new = {key(r["path"]): r["true_species"] == r["predicted_species"]
               for r in cv_predictions(EXP / "Task-B-clean14-cv", model)}
        clean[model] = new
        old = {key(r["path"]): r["true_species"] == r["predicted_species"]
               for r in cv_predictions(CV_V2, model) if r["true_species"] not in EXCLUDED}
        assert old.keys() == new.keys(), model
        a, b, p = mcnemar(new, old)
        tests.append({"comparison": "CV OOF: 14-species (new rule) vs CV v2 17-species (old rule), same 4115 images",
                      "model": model, "a": "clean14", "b": "cv_v2", "n": len(new),
                      "acc_a": np.mean(list(new.values())), "acc_b": np.mean(list(old.values())),
                      "only_a_correct": a, "only_b_correct": b, "p_exact": p})
    pairs = [("swin_tiny", "convnext_tiny"), ("convnext_tiny", "efficientnet_b3"), ("efficientnet_b3", "resnet50"),
             ("resnet50", "mobilenetv3_large")]
    for x, y in pairs:
        a, b, p = mcnemar(clean[x], clean[y])
        tests.append({"comparison": "CV OOF 14-species: model vs model, same 4115 images", "model": f"{x} vs {y}",
                      "a": x, "b": y, "n": len(clean[x]), "acc_a": np.mean(list(clean[x].values())),
                      "acc_b": np.mean(list(clean[y].values())), "only_a_correct": a, "only_b_correct": b,
                      "p_exact": p})
    for x, y in pairs:
        a_all, b_all = {}, {}
        for seed in SEEDS:
            for name, store in ((x, a_all), (y, b_all)):
                for r in final_predictions(name, seed):
                    store[(seed, key(r["path"]))] = r["true_species"] == r["predicted_species"]
        a, b, p = mcnemar(a_all, b_all)
        tests.append({"comparison": "Final-B test, seed-matched, pooled over 3 seeds (3 x 432)", "model": f"{x} vs {y}",
                      "a": x, "b": y, "n": len(a_all), "acc_a": np.mean(list(a_all.values())),
                      "acc_b": np.mean(list(b_all.values())), "only_a_correct": a, "only_b_correct": b,
                      "p_exact": p})
    write_csv("paired_tests.csv", tests)
    return tests


def august_vs_clean14():
    august = defaultdict(dict)
    for r in read_csv(AUGUST):
        if r["experiment"] == "main" and r["split"] == "test" and r["true_class"] not in EXCLUDED:
            august[r["backbone"]][(int(r["seed"]), key(r["image_path"]))] = r["predicted_class"] == r["true_class"]
    rows, tests = [], []
    for model in MODELS:
        new = {(seed, key(r["path"])): r["true_species"] == r["predicted_species"]
               for seed in SEEDS for r in final_predictions(model, seed)}
        old = august[model]
        assert old.keys() == new.keys(), (model, len(old), len(new))
        per_seed = lambda d: [100 * np.mean([v for (s, _), v in d.items() if s == seed]) for seed in SEEDS]
        a, b, p = mcnemar(new, old)
        rows.append({"model": model,
                     "august_17sp_acc_on_14sp_test": statistics.mean(per_seed(old)),
                     "august_per_seed": " / ".join(f"{x:.2f}" for x in per_seed(old)),
                     "clean14_acc": statistics.mean(per_seed(new)),
                     "clean14_per_seed": " / ".join(f"{x:.2f}" for x in per_seed(new)),
                     "difference_pts": statistics.mean(per_seed(new)) - statistics.mean(per_seed(old)),
                     "only_clean14_correct": a, "only_august_correct": b, "p_exact": p})
    write_csv("august_vs_clean14.csv", rows)
    return rows


def origin_map():
    records = load_origin_samples(CACHE, PHASE1_MANIFEST, ("train", "val", "test"))
    return {key(r["path"]): r["origin_key"] for r in records}


def errors(origins):
    by_origin = defaultdict(Counter)
    size = Counter(origins.values())
    confusions = Counter()
    for model in MODELS:
        for r in cv_predictions(EXP / "Task-B-clean14-cv", model):
            if r["true_species"] != r["predicted_species"]:
                by_origin[r["origin_key"]][model] += 1
                confusions[("cv_oof", r["true_species"], r["predicted_species"])] += 1
    rows = []
    for origin, counts in by_origin.items():
        rows.append({"origin_key": origin, "n_images": size[origin], **{m: counts[m] for m in MODELS},
                     "total_errors_5_models": sum(counts.values()),
                     "share_of_images_wrong_mean": sum(counts.values()) / (5 * size[origin])})
    rows.sort(key=lambda r: -r["total_errors_5_models"])
    write_csv("oof_errors_by_origin.csv", rows)

    missed = defaultdict(list)
    test_origins = Counter()
    for model in MODELS:
        for seed in SEEDS:
            for r in final_predictions(model, seed):
                k = key(r["path"])
                test_origins[origins[k]] += 1
                if r["true_species"] != r["predicted_species"]:
                    missed[k].append(f"{model}:s{seed}->{r['predicted_species']}")
                    confusions[("final_test", r["true_species"], r["predicted_species"])] += 1
    final_rows = sorted(({"image": k, "origin_key": origins[k], "n_runs_wrong_of_15": len(v), "runs": "; ".join(v)}
                         for k, v in missed.items()), key=lambda r: (-r["n_runs_wrong_of_15"], r["image"]))
    write_csv("final_b_test_errors.csv", final_rows)
    write_csv("confusions.csv", [{"source": s, "true": t, "predicted": p, "count": c}
                                 for (s, t, p), c in confusions.most_common()])
    return rows, final_rows, confusions, {o: n // 15 for o, n in test_origins.items()}


def degraded():
    per = {}
    for variant, root in (("rgb", "Task-B-clean14-cv"), ("grayscale", "Task-B-clean14-cv-grayscale"),
                          ("px16", "Task-B-clean14-cv-px16")):
        rows = cv_predictions(EXP / root, "convnext_tiny")
        classes = sorted({r["true_species"] for r in rows})
        f1 = f1_score([r["true_species"] for r in rows], [r["predicted_species"] for r in rows],
                      labels=classes, average=None)
        per[variant] = dict(zip(classes, f1))
    rows = [{"species": s, "f1_rgb": per["rgb"][s], "f1_grayscale": per["grayscale"][s], "f1_px16": per["px16"][s],
             "drop_grayscale": per["rgb"][s] - per["grayscale"][s], "drop_px16": per["rgb"][s] - per["px16"][s]}
            for s in per["rgb"]]
    rows.sort(key=lambda r: -r["drop_grayscale"])
    write_csv("degraded_per_species.csv", rows)
    return rows


def curves():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for col, (label, pattern) in enumerate((("CV, mean over 5 folds", "Task-B-clean14-cv/{m}/fold_*"),
                                            ("Final-B, mean over 3 seeds", "Task-B-clean14-final/seed_*/{m}"))):
        for model in MODELS:
            runs = [json.loads(Path(p, "history.json").read_text())
                    for p in glob.glob(str(EXP / pattern.format(m=model)))]
            for row, metric in enumerate(("val_loss", "val_accuracy")):
                values = defaultdict(list)
                for h in runs:
                    for r in h:
                        values[r["epoch"]].append(r[metric])
                epochs = sorted(values)
                axes[row, col].plot(epochs, [np.mean(values[e]) for e in epochs], label=model)
        axes[0, col].set_title(label)
        axes[1, col].set_xlabel("epoch")
    axes[0, 0].set_ylabel("val loss (label smoothing 0.1)")
    axes[1, 0].set_ylabel("val accuracy")
    for ax in axes[1]:
        ax.set_ylim(0.9, 1.0)
    for ax in axes.flat:
        ax.axvline(5.5, color="gray", ls=":", lw=1)
        ax.grid(alpha=0.3)
    axes[0, 1].legend(fontsize=8)
    fig.suptitle("Mean over runs; a run that stopped early contributes only to its epochs. Dotted line: end of warmup.",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "curves.png", dpi=150)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n, bad, commits, torches, data = integrity()
    print(f"Integrity: {n} runs, mismatches {len(bad)}, commits {commits}, torch {torches}, data {data}")
    conv = convergence()
    print("\nRuns that reached epoch 50:")
    for r in conv:
        if r["reached_cap"]:
            print(f"  {r['run']}: best {r['best_epoch']}, val acc 40 -> best "
                  f"{100 * r['val_acc_epoch40']:.2f} -> {100 * r['val_acc_best']:.2f}, "
                  f"val loss gain after 40 {r['val_loss_gain_after_epoch40']:.4f}, lr@45 {r['lr_at_epoch45']:.1e}")
    print("\nPaired tests:")
    for t in paired_tests():
        print(f"  {t['comparison'][:40]:40s} {t['model']:36s} {100 * t['acc_a']:.2f} vs {100 * t['acc_b']:.2f} "
              f"(+{t['only_a_correct']}/-{t['only_b_correct']}) p={t['p_exact']:.3g}")
    print("\nAugust Final-B (17 species) vs 14-species Final-B on the same 432 test images:")
    for r in august_vs_clean14():
        print(f"  {r['model']:18s} {r['august_17sp_acc_on_14sp_test']:.2f} ({r['august_per_seed']}) -> "
              f"{r['clean14_acc']:.2f} ({r['clean14_per_seed']}) p={r['p_exact']:.3g}")
    oof, final, confusions, test_origin_sizes = errors(origin_map())
    print("\nCV OOF errors by origin (top 8):")
    for r in oof[:8]:
        print(f"  {r['origin_key']:55s} n={r['n_images']:3d} errors(5 models)={r['total_errors_5_models']:3d} "
              + " ".join(str(r[m]) for m in MODELS))
    total = sum(r["total_errors_5_models"] for r in oof)
    print(f"  total OOF errors over 5 models: {total}")
    print("\nFinal-B test images wrong in >= 3 of 15 runs:")
    for r in final:
        if r["n_runs_wrong_of_15"] >= 3:
            print(f"  {r['n_runs_wrong_of_15']:2d}/15 {r['image']}")
    print(f"  distinct wrong test images: {len(final)}; test origins: {len(test_origin_sizes)}")
    print("\nTop confusions:")
    for (source, t, p), c in confusions.most_common(10):
        print(f"  {source:10s} {t} -> {p}: {c}")
    print("\nConvNeXt OOF F1 per species (rgb / grayscale / px16), largest grayscale drop first:")
    for r in degraded():
        print(f"  {r['species']:40s} {r['f1_rgb']:.3f} / {r['f1_grayscale']:.3f} / {r['f1_px16']:.3f}")
    curves()
    print(f"\nWritten to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
