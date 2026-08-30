"""
Dung confusion matrix 17x17 cho paper (person 3).

Nguon duy nhat cua so lieu: predictions.csv trong registry.
confusion_matrices.npz CHI dung de doi chieu count, khong dung de ve.

Sinh ra:
    artifact/person3/confusion/fig_confusion.pdf
    artifact/person3/confusion/fig_confusion.png          (300 DPI)
    artifact/person3/confusion/confusion_convnext_main.csv
    artifact/person3/confusion/confusion_metadata.json

Tieu chi chon model (cong bo TRUOC khi nhin ket qua confusion):
    backbone co macro-F1 mean cao nhat tren test, trung binh 3 main seed.
    Tieu chi nay duoc TINH tu main_test_summary.csv, khong hard-code.
    KHONG chon rieng run co accuracy cao nhat lam dai dien.

Cach gop 3 seed:
    Cong count cua 3 confusion matrix seed (42, 123, 456) thanh mot aggregate.
    Tong count aggregate = 532 x 3 = 1596. Mau so khi row-normalize la tong
    count cua ca 3 seed, tuc moi seed dong gop deu nhau (moi seed du doan dung
    532 anh test).

Usage:
    .venv/bin/python scripts/person3/make_confusion_and_perclass.py
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_REGISTRY = (
    ROOT
    / "docs/reports/experiments/phase_1b/final_b_new_recomputed"
    / "20260819T043733Z"
)
DEFAULT_OUT_DIR = ROOT / "artifact/person3/confusion"

MAIN_SEEDS = ("42", "123", "456")
SELECTION_CRITERION = "highest macro-F1 mean on the test split, averaged over the 3 main seeds"
SELECTION_COLUMN = "f1_macro_mean"

BACKBONE_LABEL = {
    "resnet50": "ResNet-50",
    "efficientnet_b3": "EfficientNet-B3",
    "mobilenetv3_large": "MobileNetV3-Large",
    "convnext_tiny": "ConvNeXt-Tiny",
    "swin_tiny": "Swin-Tiny",
}

# Ho cua tung loai trong tap 17 lop -- QUY UOC 5 HO (phan loai hoc dung).
# Araucaria araucana thuoc Araucariaceae, KHONG gop vao Pinaceae.
# Luu y: docs/references/WoodID_plan.md muc 3.2 dung so do 4 ho cho Task A
# (gop Araucaria vao Pinaceae). Bai nay dung 5 ho de khop voi hinh Grad-CAM
# va voi phan loai hoc; moi ty le theo ho deu tinh lai tren so do 5 ho.
SPECIES_FAMILY = {
    "Cunninghamia lanceolata": "Cupressaceae",
    "Callitris columellaris": "Cupressaceae",
    "Fokienia hodginsii": "Cupressaceae",
    "Cryptomeria japonica": "Cupressaceae",
    "Xanthocyparis vietnamensis": "Cupressaceae",
    "Chamaecyparis obtusa": "Cupressaceae",
    "Calocedrus rupestris": "Cupressaceae",
    "Thujopsis dolabrata": "Cupressaceae",
    "Araucaria araucana": "Araucariaceae",
    "Pinus dalatensis": "Pinaceae",
    "Pinus radiata": "Pinaceae",
    "Pinus krempfii": "Pinaceae",
    "Picea glauca": "Pinaceae",
    "Pseudotsuga brevifolia": "Pinaceae",
    "Pinus cernua": "Pinaceae",
    "Podocarpus pilgeri": "Podocarpaceae",
    "Taxus wallichiana": "Taxaceae",
}
FAMILY_ORDER = [
    "Araucariaceae",
    "Cupressaceae",
    "Pinaceae",
    "Podocarpaceae",
    "Taxaceae",
]

# Sequential blue ramp 100->700 (dataviz skill, references/palette.md).
# Mot hue, light->dark, dung cho magnitude. Buoc dau la mau nen chart.
SEQUENTIAL_BLUE = [
    "#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec",
    "#3987e5", "#256abf", "#184f95", "#0d366b",
]
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
CMAP = LinearSegmentedColormap.from_list("wood_blue", SEQUENTIAL_BLUE)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"[ERROR] khong tim thay {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def select_backbone(summary_rows) -> tuple[str, dict]:
    """Ap dung tieu chi da cong bo truoc, khong hard-code ket qua."""
    ranking = sorted(
        summary_rows, key=lambda r: float(r[SELECTION_COLUMN]), reverse=True
    )
    winner = ranking[0]
    table = [
        {
            "backbone": r["backbone"],
            SELECTION_COLUMN: float(r[SELECTION_COLUMN]),
            "accuracy_mean": float(r["accuracy_mean"]),
        }
        for r in ranking
    ]
    return winner["backbone"], {"criterion": SELECTION_CRITERION, "ranking": table}


def order_species(names) -> list[str]:
    return sorted(
        names,
        key=lambda s: (
            FAMILY_ORDER.index(SPECIES_FAMILY[s])
            if SPECIES_FAMILY.get(s) in FAMILY_ORDER
            else len(FAMILY_ORDER),
            s,
        ),
    )


def build_matrix(preds, classes) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Tra ve (aggregate 3 seed, {seed: matrix rieng})."""
    idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    per_seed = {s: np.zeros((n, n), dtype=np.int64) for s in MAIN_SEEDS}
    for p in preds:
        per_seed[p["seed"]][idx[p["true_class"]], idx[p["predicted_class"]]] += 1
    agg = sum(per_seed.values())
    return agg, per_seed


def crosscheck_npz(registry: Path, backbone: str, per_seed, classes, npz_classes) -> dict:
    """Doi chieu count voi confusion_matrices.npz.

    npz dung thu tu lop goc (npz_classes, alphabet) con ta sap theo family,
    nen phai HOAN VI ve cung thu tu roi so sanh TUNG PHAN TU -- khong dung
    bat bien hoan vi (tong / duong cheo / multiset row-sum), vi phep do bo
    lot moi sai lech chi la hoan vi cua nhau.
    """
    path = registry / "confusion_matrices.npz"
    if not path.exists():
        return {"status": "npz_missing"}
    z = np.load(path)
    if not npz_classes:
        return {"status": "npz_class_order_unknown"}
    perm = [npz_classes.index(c) for c in classes]      # family order -> npz order
    out = {}
    for seed, mat in per_seed.items():
        key = f"taskB17__main__seed_{seed}__{backbone}__test"
        if key not in z.files:
            out[seed] = "key_missing"
            continue
        ref = np.asarray(z[key])[np.ix_(perm, perm)]     # ve thu tu family
        diff = int((ref != mat).sum())
        out[seed] = {
            "npz_key": key,
            "elementwise_exact_match": diff == 0,
            "cells_differing": diff,
            "npz_total": int(ref.sum()),
            "rebuilt_total": int(mat.sum()),
            "comparison": "exact, after permuting npz to the family class order",
        }
    return out


def plot(agg, classes, backbone, n_seeds, out_pdf: Path, out_png: Path):
    n = len(classes)
    row_tot = agg.sum(1, keepdims=True)
    pct = np.divide(
        agg, row_tot, out=np.zeros(agg.shape, dtype=float), where=row_tot > 0
    ) * 100.0

    fig, ax = plt.subplots(figsize=(11.5, 9.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    im = ax.imshow(pct, cmap=CMAP, vmin=0, vmax=100, aspect="equal")

    # spacer 2px giua cac o
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.6)
    ax.tick_params(which="minor", length=0)

    # duong phan tach family
    bounds, prev = [], None
    for i, c in enumerate(classes):
        fam = SPECIES_FAMILY.get(c)
        if prev is not None and fam != prev:
            bounds.append(i - 0.5)
        prev = fam
    for b in bounds:
        ax.axhline(b, color=INK_PRIMARY, linewidth=0.8)
        ax.axvline(b, color=INK_PRIMARY, linewidth=0.8)

    # annotate: count aggregate; o loi khac 0 duoc vien lai cho de tim
    for i in range(n):
        for j in range(n):
            v = int(agg[i, j])
            if v == 0:
                continue
            ax.text(
                j, i, str(v),
                ha="center", va="center", fontsize=7.5, zorder=11,
                color="#ffffff" if pct[i, j] > 55 else INK_PRIMARY,
            )
            if i != j:
                # thut vao 0.06 va zorder cao: luoi trang (spacer) duoc ve
                # sau imshow nen se de len neu vien nam dung mep o
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.44, i - 0.44), 0.88, 0.88,
                        fill=False, edgecolor="#e34948", linewidth=1.0,
                        zorder=10,
                    )
                )

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8.5, color=INK_PRIMARY)
    ax.set_yticklabels(classes, fontsize=8.5, color=INK_PRIMARY)
    ax.set_xlabel("Predicted species", fontsize=10.5, color=INK_PRIMARY, labelpad=8)
    ax.set_ylabel("True species", fontsize=10.5, color=INK_PRIMARY, labelpad=8)
    for s in ax.spines.values():
        s.set_edgecolor(INK_SECONDARY); s.set_linewidth(0.8)

    ax.set_title(
        f"{BACKBONE_LABEL.get(backbone, backbone)} — Task B test confusion matrix\n"
        f"counts summed over {n_seeds} seeds ({', '.join(MAIN_SEEDS)}); "
        f"cell shading is row-normalised",
        fontsize=11.5, color=INK_PRIMARY, pad=14,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.02)
    cbar.set_label(
        "Row-normalised percentage of true-class images (%)\n"
        "cell annotations are raw counts summed over 3 seeds",
        fontsize=9, color=INK_PRIMARY,
    )
    cbar.ax.tick_params(labelsize=8.5, color=INK_SECONDARY)
    cbar.outline.set_edgecolor(INK_SECONDARY)

    # chu thich duong phan tach family
    ax.plot([], [], color=INK_PRIMARY, linewidth=1.6, label="family boundary")
    ax.plot([], [], color="#e34948", linewidth=1.4, label="off-diagonal (misclassified)")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.30), frameon=False,
        fontsize=9, labelcolor=INK_PRIMARY, ncol=2, handlelength=2.4,
    )

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    # CreationDate=None -> PDF byte-identical giua cac lan chay (kiem tra lai duoc)
    fig.savefig(out_pdf, format="pdf", facecolor=SURFACE, bbox_inches="tight",
                metadata={"CreationDate": None})
    fig.savefig(out_png, format="png", dpi=300, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)



# ==========================================================================
# PHAN 2 -- per-species F1 va phan tich loi (muc 6.3)
# ==========================================================================
PERCLASS_OUT_DIR = ROOT / "artifact/person3/per_class"


def perclass_stats(perclass_rows) -> tuple[dict, list[str], list[str], dict]:
    """mean/SD per-class F1 tren 3 main seed, cho tung backbone."""
    test = [
        r for r in perclass_rows
        if r["split"] == "test" and r["experiment"] == "main"
    ]
    if not test:
        raise SystemExit("[ERROR] per_class_metrics khong co dong main/test")

    vals = defaultdict(list)          # (species, backbone) -> [f1 tung seed]
    support = {}
    for r in test:
        vals[(r["class_name"], r["backbone"])].append(float(r["f1"]))
        support[r["class_name"]] = int(r["support"])

    counts = {len(v) for v in vals.values()}
    if counts != {len(MAIN_SEEDS)}:
        raise SystemExit(f"[ERROR] so seed khong dong deu giua (loai, backbone): {counts}")

    backbones = [b for b in BACKBONE_LABEL if b in {r["backbone"] for r in test}]
    species = order_species(support)

    stats = {}
    for (sp, b), v in vals.items():
        stats[(sp, b)] = {
            "mean": statistics.fmean(v),
            "sd": statistics.stdev(v),
            "n_seeds": len(v),
        }
    run_ids = sorted({r["run_id"] for r in test})
    return stats, species, backbones, {"support": support, "run_ids": run_ids}


def error_breakdown(preds) -> dict:
    """Phan tich loi truc tiep tu predictions.csv (test + main)."""
    rows = [p for p in preds if p["split"] == "test" and p["experiment"] == "main"]
    backbones = [b for b in BACKBONE_LABEL if b in {p["backbone"] for p in rows}]

    pairs = defaultdict(Counter)      # backbone -> Counter[(true, pred)]
    fam_tot, fam_err = Counter(), Counter()
    fam_tot_nr, fam_err_nr = Counter(), Counter()
    for p in rows:
        fam = SPECIES_FAMILY[p["true_class"]]
        fam_tot[fam] += 1
        if p["backbone"] != "resnet50":
            fam_tot_nr[fam] += 1
        if p["correct"] == "False":
            pairs[p["backbone"]][(p["true_class"], p["predicted_class"])] += 1
            fam_err[fam] += 1
            if p["backbone"] != "resnet50":
                fam_err_nr[fam] += 1

    def fam_table(err, tot):
        return {
            f: {
                "errors": err[f],
                "predictions": tot[f],
                "error_rate_pct": round(err[f] / tot[f] * 100, 4) if tot[f] else None,
            }
            for f in FAMILY_ORDER
        }

    # cap nham gop tren moi backbone
    total_pairs = Counter()
    for c in pairs.values():
        total_pairs.update(c)

    per_pair_by_backbone = {}
    for (t, pr), n in total_pairs.most_common():
        per_pair_by_backbone[f"{t} -> {pr}"] = {
            "total_over_all_backbones": n,
            "by_backbone": {b: pairs[b].get((t, pr), 0) for b in backbones},
        }

    return {
        "source": "predictions.csv (split=test, experiment=main)",
        "n_predictions": len(rows),
        "n_errors": sum(sum(c.values()) for c in pairs.values()),
        "errors_by_backbone": {b: sum(pairs[b].values()) for b in backbones},
        "family_error_rate_all_backbones": fam_table(fam_err, fam_tot),
        "family_error_rate_excluding_resnet50": fam_table(fam_err_nr, fam_tot_nr),
        "cross_family_errors": sum(
            1 for p in rows
            if p["correct"] == "False"
            and SPECIES_FAMILY[p["true_class"]] != SPECIES_FAMILY[p["predicted_class"]]
        ),
        "within_family_errors": sum(
            1 for p in rows
            if p["correct"] == "False"
            and SPECIES_FAMILY[p["true_class"]] == SPECIES_FAMILY[p["predicted_class"]]
        ),
        "confusion_pairs": per_pair_by_backbone,
    }


def plot_perclass(stats, species, backbones, support, out_pdf: Path, out_png: Path):
    """Small multiples: 1 panel / backbone. >3 series so sanh all-pairs thi
    facet thay vi cap them mau (dataviz skill)."""
    n_sp = len(species)
    y = np.arange(n_sp)[::-1]          # loai dau tien nam tren cung

    fig, axes = plt.subplots(
        1, len(backbones), figsize=(16.5, 8.2), sharey=True,
        gridspec_kw={"wspace": 0.10},
    )
    fig.patch.set_facecolor(SURFACE)

    # bien giua cac family (theo toa do y da dao nguoc)
    bounds, prev = [], None
    for i, c in enumerate(species):
        fam = SPECIES_FAMILY.get(c)
        if prev is not None and fam != prev:
            bounds.append(y[i] + 0.5)
        prev = fam

    xmin = 50.0
    for ax, b in zip(axes, backbones):
        ax.set_facecolor(SURFACE)
        means = np.array([stats[(s, b)]["mean"] for s in species]) * 100
        sds = np.array([stats[(s, b)]["sd"] for s in species]) * 100

        # ray dan dong (dai deu, khong ma hoa do lon) -- dot plot ma hoa
        # bang VI TRI nen truc x duoc phep khong bat dau tu 0
        ax.hlines(y, xmin, 103, color="#f0f0ee", linewidth=6.0, zorder=0)
        ax.errorbar(
            means, y, xerr=sds, fmt="none",
            ecolor="#6da7ec", elinewidth=2.0, capsize=3.0, capthick=1.6, zorder=2,
        )
        ax.scatter(means, y, s=42, color="#2a78d6", zorder=3,
                   edgecolor=SURFACE, linewidth=1.2)

        for bnd in bounds:
            ax.axhline(bnd, color=INK_SECONDARY, linewidth=1.0, linestyle="-", zorder=0)

        ax.set_xlim(xmin, 103)
        ax.set_ylim(-0.8, n_sp - 0.2)
        ax.set_xticks([50, 60, 70, 80, 90, 100])
        ax.tick_params(axis="x", labelsize=8.5, colors=INK_SECONDARY)
        ax.set_title(BACKBONE_LABEL.get(b, b), fontsize=10.5, color=INK_PRIMARY, pad=8)
        ax.set_xlabel("F1 (%)", fontsize=9.5, color=INK_SECONDARY)
        ax.grid(axis="x", color="#e8e8e6", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(INK_SECONDARY)
        ax.spines["bottom"].set_linewidth(0.8)

        # nhan truc tiep cho gia tri thap (de tim outlier ma khong phai do truc)
        for yi, m, sd in zip(y, means, sds):
            if m >= 90:
                continue
            left = m - sd - 1.6
            if left > xmin + 3.0:                 # du cho ben trai
                ax.text(left, yi, f"{m:.1f}", ha="right", va="center",
                        fontsize=7.5, color=INK_PRIMARY)
            else:                                  # sat mep -> lat sang phai
                ax.text(m + sd + 1.6, yi, f"{m:.1f}", ha="left", va="center",
                        fontsize=7.5, color=INK_PRIMARY)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(species, fontsize=9, color=INK_PRIMARY)
    axes[0].tick_params(axis="y", length=0)

    # nhan family o le trai
    prev, start = None, 0
    for i, c in enumerate(species + [None]):
        fam = SPECIES_FAMILY.get(c) if c else None
        if prev is not None and fam != prev:
            mid = (y[start] + y[i - 1]) / 2
            axes[0].annotate(
                prev, xy=(0, mid), xycoords=("axes fraction", "data"),
                xytext=(-190, 0), textcoords="offset points",
                ha="left", va="center", fontsize=9.5, style="italic",
                color=INK_SECONDARY, annotation_clip=False,
            )
            start = i
        prev = fam

    fig.suptitle(
        "Per-species F1 on the Task B test set, by architecture\n"
        f"dot = mean over {len(MAIN_SEEDS)} seeds ({', '.join(MAIN_SEEDS)}); "
        "bar = standard deviation across those seeds, not an interval estimate",
        fontsize=12, color=INK_PRIMARY, y=0.99,
    )
    fig.tight_layout(rect=[0.10, 0.02, 1, 0.93])
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    # CreationDate=None -> PDF byte-identical giua cac lan chay (kiem tra lai duoc)
    fig.savefig(out_pdf, format="pdf", facecolor=SURFACE, bbox_inches="tight",
                metadata={"CreationDate": None})
    fig.savefig(out_png, format="png", dpi=300, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def run_perclass(reg: Path, out: Path, meta: dict, preds_all) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pc_path = reg / "per_class_metrics.csv"
    preds_path = reg / "predictions.csv"

    stats, species, backbones, extra = perclass_stats(read_csv(pc_path))
    support, pc_run_ids = extra["support"], extra["run_ids"]

    # --- CSV ---
    csv_out = out / "perclass_f1_summary.csv"
    with csv_out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        header = ["species", "family", "n_test_images"]
        for b in backbones:
            header += [f"{b}_f1_mean_pct", f"{b}_f1_sd_pct"]
        w.writerow(header)
        for s in species:
            row = [s, SPECIES_FAMILY.get(s, ""), support[s]]
            for b in backbones:
                row += [
                    round(stats[(s, b)]["mean"] * 100, 4),
                    round(stats[(s, b)]["sd"] * 100, 4),
                ]
            w.writerow(row)
    print(f"[OK] {rel(csv_out)}")

    pdf, png = out / "fig_perclass_f1.pdf", out / "fig_perclass_f1.png"
    plot_perclass(stats, species, backbones, support, pdf, png)
    print(f"[OK] {rel(pdf)}")
    print(f"[OK] {rel(png)} (300 DPI)")

    errors = error_breakdown(preds_all)
    pred_run_ids = sorted({
        p["run_id"] for p in preds_all
        if p["split"] == "test" and p["experiment"] == "main"
    })

    metadata = {
        "generated_by": "scripts/person3/make_confusion_and_perclass.py",
        "source_files": {
            "per_class_f1": rel(pc_path),
            "error_analysis": rel(preds_path),
        },
        "registry_dir": rel(reg),
        "registry_created_at_utc": meta.get("registry_created_at_utc"),
        "split_version": meta.get("split_version"),
        "seeds": list(MAIN_SEEDS),
        "dispersion_reported_as": (
            "standard deviation over 3 seeds (not an interval estimate)"
        ),
        "backbones": backbones,
        "class_order": species,
        "class_family": {s: SPECIES_FAMILY.get(s) for s in species},
        "support": support,
        "per_class_f1": {
            s: {
                b: {
                    "mean_pct": round(stats[(s, b)]["mean"] * 100, 4),
                    "sd_pct": round(stats[(s, b)]["sd"] * 100, 4),
                }
                for b in backbones
            }
            for s in species
        },
        "error_analysis": errors,
        "source_run_ids": {
            "per_class_metrics_csv": pc_run_ids,
            "predictions_csv": pred_run_ids,
        },
    }
    meta_out = out / "perclass_metadata.json"
    meta_out.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[OK] {rel(meta_out)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry-dir", type=Path, default=DEFAULT_REGISTRY)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--perclass-out-dir", type=Path, default=PERCLASS_OUT_DIR)
    args = ap.parse_args()

    reg, out = args.registry_dir, args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    meta_path = reg / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    summary = read_csv(reg / "main_test_summary.csv")
    backbone, selection = select_backbone(summary)
    print(f"[SELECT] {SELECTION_CRITERION}")
    for r in selection["ranking"]:
        mark = " <-- chon" if r["backbone"] == backbone else ""
        print(f"    {r['backbone']:<19} macro-F1 {r[SELECTION_COLUMN]*100:.3f}{mark}")

    preds_path = reg / "predictions.csv"
    preds_all = read_csv(preds_path)
    preds = [
        p for p in preds_all
        if p["split"] == "test" and p["experiment"] == "main" and p["backbone"] == backbone
    ]
    if not preds:
        raise SystemExit(f"[ERROR] khong co prediction test/main cho {backbone}")

    seeds_found = sorted({p["seed"] for p in preds})
    if seeds_found != sorted(MAIN_SEEDS):
        raise SystemExit(f"[ERROR] seed khong khop: {seeds_found} != {sorted(MAIN_SEEDS)}")

    classes = order_species({p["true_class"] for p in preds})
    agg, per_seed = build_matrix(preds, classes)

    # --- kiem tra bat buoc: tong count = 532 x so seed ---
    n_test = meta.get("split_counts", {}).get("test", 532)
    expected = n_test * len(MAIN_SEEDS)
    if int(agg.sum()) != expected:
        raise SystemExit(f"[ERROR] tong count {int(agg.sum())} != {n_test} x {len(MAIN_SEEDS)} = {expected}")
    for s, m in per_seed.items():
        if int(m.sum()) != n_test:
            raise SystemExit(f"[ERROR] seed {s}: tong count {int(m.sum())} != {n_test}")
    stacked = np.sum([per_seed[s] for s in MAIN_SEEDS], axis=0)
    if not np.array_equal(stacked, agg):
        raise SystemExit("[ERROR] aggregate khong bang tong element-wise cua 3 ma tran seed")
    print(f"[CHECK] tong count aggregate = {int(agg.sum())} = {n_test} x {len(MAIN_SEEDS)}  OK")
    print(f"[CHECK] aggregate == tong element-wise cua {len(MAIN_SEEDS)} ma tran seed  OK")

    xcheck = crosscheck_npz(
        reg, backbone, per_seed, classes, meta.get("classes", [])
    )
    for s, v in xcheck.items() if isinstance(xcheck, dict) else []:
        if isinstance(v, dict):
            ok = v["elementwise_exact_match"]
            print(f"[NPZ ] seed {s}: {'khop tung o' if ok else 'LECH'} "
                  f"({v['cells_differing']} o khac, npz={v['npz_total']})")

    # --- CSV count aggregate ---
    csv_out = out / "confusion_convnext_main.csv"
    with csv_out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["true_class", "family", *classes, "row_total"])
        for i, c in enumerate(classes):
            w.writerow([c, SPECIES_FAMILY.get(c, ""), *agg[i].tolist(), int(agg[i].sum())])
    print(f"[OK] {rel(csv_out)}")

    per_seed_csv = out / "confusion_convnext_main_per_seed.csv"
    with per_seed_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "true_class", "family", *classes, "row_total"])
        for s in MAIN_SEEDS:
            m = per_seed[s]
            for i, c in enumerate(classes):
                w.writerow([s, c, SPECIES_FAMILY.get(c, ""), *m[i].tolist(), int(m[i].sum())])
    print(f"[OK] {rel(per_seed_csv)} (3 ma tran rieng, cong lai ra file aggregate)")

    pdf, png = out / "fig_confusion.pdf", out / "fig_confusion.png"
    plot(agg, classes, backbone, len(MAIN_SEEDS), pdf, png)
    print(f"[OK] {rel(pdf)}")
    print(f"[OK] {rel(png)} (300 DPI)")

    errors = [
        {
            "true_class": classes[i],
            "predicted_class": classes[j],
            "count_over_3_seeds": int(agg[i, j]),
        }
        for i in range(len(classes))
        for j in range(len(classes))
        if i != j and agg[i, j] > 0
    ]
    errors.sort(key=lambda e: -e["count_over_3_seeds"])

    metadata = {
        "generated_by": "scripts/person3/make_confusion_and_perclass.py",
        "source_file": rel(preds_path),
        "crosscheck_only_source": rel(reg / "confusion_matrices.npz"),
        "registry_dir": rel(reg),
        "registry_created_at_utc": meta.get("registry_created_at_utc"),
        "split_version": meta.get("split_version"),
        "model_selection": selection,
        "selected_backbone": backbone,
        "aggregation": {
            "method": "element-wise sum of the per-seed 17x17 count matrices",
            "seeds": list(MAIN_SEEDS),
            "n_seeds": len(MAIN_SEEDS),
            "images_per_seed": n_test,
            "total_count": int(agg.sum()),
            "expected_total": expected,
            "row_normalisation_denominator": "row total of the summed counts (all 3 seeds)",
            "per_seed_matrices_file": "confusion_convnext_main_per_seed.csv",
            "verification": (
                "the aggregate was checked element-wise against the sum of the "
                "three per-seed matrices, not only against the grand total"
            ),
        },
        "class_order": classes,
        "class_family": {c: SPECIES_FAMILY.get(c) for c in classes},
        "source_run_ids": sorted({p["run_id"] for p in preds}),
        "npz_crosscheck": xcheck,
        "per_seed_totals": {s: int(m.sum()) for s, m in per_seed.items()},
        "per_seed_matrices": {s: per_seed[s].tolist() for s in MAIN_SEEDS},
        "aggregate_matrix": agg.tolist(),
        "aggregate_equals_elementwise_sum": True,
        "total_errors_over_3_seeds": int(agg.sum() - np.trace(agg)),
        "confusion_pairs": errors,
    }
    meta_out = out / "confusion_metadata.json"
    meta_out.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] {rel(meta_out)}")

    print("\n=== per-species F1 + phan tich loi (6.3) ===")
    run_perclass(reg, args.perclass_out_dir, meta, preds_all)


if __name__ == "__main__":
    main()
