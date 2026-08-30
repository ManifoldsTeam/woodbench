"""
Sinh cac bang ket qua LaTeX cho paper (person 3) tu artifact registry da recompute.

Sinh ra:
    artifact/person3/tables/tab_taskb.tex           <- main_test_summary.csv
    artifact/person3/tables/tab_ablation.tex        <- ablation_vs_main_seed42.csv
    artifact/person3/tables/tab_perspecies_full.tex <- per_class_metrics.csv
    artifact/person3/tables/tables_metadata.json    <- lineage cho ca 3 bang

Quy uoc bao cao (bat buoc, xem check_forbidden_language):
  - Main benchmark: mean +/- SD tren 3 seed. KHONG goi la 95% CI.
  - Ablation: chi 1 seed (42) -> khong co +/- , chi la diem don.
  - Khong dung tu "significant" / "statistically significant" o bat ky dau.

Usage:
    python scripts/person3/make_result_tables.py
    python scripts/person3/make_result_tables.py --registry-dir <duong/dan/khac>
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_REGISTRY = (
    ROOT
    / "docs/reports/experiments/phase_1b/final_b_new_recomputed"
    / "20260819T043733Z"
)
DEFAULT_OUT_DIR = ROOT / "artifact/person3/tables"

MAIN_SEEDS = ("42", "123", "456")
ABLATION_BASELINE_SEED = "42"

# Thu tu trinh bay trong paper (CNN co dien -> hieu qua -> modern)
BACKBONE_ORDER = [
    "resnet50",
    "efficientnet_b3",
    "mobilenetv3_large",
    "convnext_tiny",
    "swin_tiny",
]

BACKBONE_LABEL = {
    "resnet50": "ResNet-50",
    "efficientnet_b3": "EfficientNet-B3",
    "mobilenetv3_large": "MobileNetV3-Large",
    "convnext_tiny": "ConvNeXt-Tiny",
    "swin_tiny": "Swin-Tiny",
}

# Thu tu + nhan cho cac ablation (baseline la main seed 42 cung backbone)
ABLATION_ORDER = [
    "taskB_90rotation",
    "taskB_label_smoothing_0.0",
    "taskB_label_smoothing_0.2",
]

ABLATION_LABEL = {
    "taskB_90rotation": r"Discrete $90^\circ$ rotation",
    "taskB_label_smoothing_0.0": r"Label smoothing $0.0$",
    "taskB_label_smoothing_0.2": r"Label smoothing $0.2$",
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

# Ngon ngu bi cam trong moi output (yeu cau 6.1)
FORBIDDEN = [
    "significant",
    "95% ci",
    "95\\% ci",
    "confidence interval",
]


# --------------------------------------------------------------------------
# doc du lieu
# --------------------------------------------------------------------------
def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"[ERROR] khong tim thay {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"[ERROR] {path} khong co du lieu")
    return rows


def read_metadata(registry: Path) -> dict:
    path = registry / "metadata.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def verify_seeds(found: set[str], where: str) -> list[str]:
    """Danh sach seed phai lay TU DU LIEU, khong tin hang so.
    Caption khang dinh '3 seeds (42, 123, 456)' nen neu registry doi seed
    ma caption van in seed cu thi bai bao se noi sai."""
    got = sorted(found, key=int)
    if got != sorted(MAIN_SEEDS, key=int):
        raise SystemExit(
            f"[ERROR] {where}: seed trong du lieu {got} != MAIN_SEEDS "
            f"{sorted(MAIN_SEEDS, key=int)}. Sua MAIN_SEEDS truoc khi sinh bang."
        )
    return got


def order_backbones(keys) -> list[str]:
    known = [b for b in BACKBONE_ORDER if b in keys]
    return known + sorted(k for k in keys if k not in BACKBONE_ORDER)


# --------------------------------------------------------------------------
# lineage header dung chung cho ca 3 bang
# --------------------------------------------------------------------------
def lineage_header(source: Path, registry_ts: str, run_ids: list[str]) -> list[str]:
    lines = [
        "% AUTO-GENERATED bang scripts/person3/make_result_tables.py -- khong sua tay",
        f"% Source file      : {rel(source)}",
        f"% Registry created : {registry_ts}",
        f"% Source run ids   : {len(run_ids)} runs",
    ]
    lines += [f"%   {rid}" for rid in run_ids]
    return lines


def fmt_pm(mean: float, std: float, precision: int, bold: bool = False) -> str:
    body = f"{mean * 100:.{precision}f} \\pm {std * 100:.{precision}f}"
    return f"$\\mathbf{{{body}}}$" if bold else f"${body}$"


def fmt_delta(value: float, precision: int) -> str:
    """Delta tinh bang diem phan tram, luon co dau."""
    return f"${value * 100:+.{precision}f}$"


# --------------------------------------------------------------------------
# Bang 1 -- main benchmark (mean +/- SD tren 3 seed)
# --------------------------------------------------------------------------
def build_taskb(rows, meta, source, precision, lang="en") -> tuple[str, list[str]]:
    by_backbone = {r["backbone"]: r for r in rows}
    order = order_backbones(by_backbone)

    cols = [
        (r"Top-1 Accuracy (\%)", "accuracy_mean", "accuracy_std"),
        (r"Macro-F1 (\%)", "f1_macro_mean", "f1_macro_std"),
    ]
    best = {c: max(float(by_backbone[b][c]) for b in order) for _, c, _ in cols}

    run_ids = sorted(
        {rid for r in rows for rid in r["source_run_ids"].split("|") if rid}
    )
    seeds = verify_seeds(
        {rid.split("seed_")[1].split("__")[0] for rid in run_ids if "seed_" in rid},
        "main_test_summary.csv",
    )
    width = max(len(BACKBONE_LABEL.get(b, b)) for b in order)

    body = []
    for b in order:
        r = by_backbone[b]
        cells = [
            fmt_pm(
                float(r[m]), float(r[s]), precision, float(r[m]) == best[m]
            )
            for _, m, s in cols
        ]
        label = BACKBONE_LABEL.get(b, b)
        body.append(f"{label:<{width}} & " + " & ".join(cells) + r" \\")

    n_classes = len(meta.get("classes", [])) or 17
    n_test = meta.get("split_counts", {}).get("test")
    n_runs = rows[0].get("runs", str(len(seeds)))

    caption = (
        f"Species-level classification performance on the Task B test set\n"
        f"({n_classes} species{f', {n_test} images' if n_test else ''}, partitioned by origin key\n"
        f"so that images sharing an origin key fall in a single split).\n"
        f"Values are the mean $\\pm$ standard deviation across {n_runs} training seeds\n"
        f"({', '.join(seeds)}); the spread is reported as a standard deviation over\n"
        f"three runs and is not an interval estimate. Best value per column in bold."
    )

    tex = "\n".join(
        [
            "% Table: Task B main benchmark (species-level, 17 classes)",
            *lineage_header(source, meta.get("registry_created_at_utc", "n/a"), run_ids),
            r"\begin{table}[!htbp]",
            r"\centering",
            r"\caption{" + caption + "}",
            r"\label{tab:task_b}",
            r"\begin{tabular}{l" + "c" * len(cols) + "}",
            r"\toprule",
            r"\textbf{Architecture} & "
            + " & ".join(rf"\textbf{{{h}}}" for h, _, _ in cols)
            + r" \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return tex, run_ids


# --------------------------------------------------------------------------
# Bang 2 -- ablation vs main seed 42 (cung backbone), 1 seed duy nhat
# --------------------------------------------------------------------------
def build_ablation(rows, meta, source, precision, lang="en") -> tuple[str, list[str]]:
    test = [r for r in rows if r["split"] == "test"]
    if not test:
        raise SystemExit("[ERROR] ablation csv khong co dong split=test")

    seeds = sorted({r["seed"] for r in test})
    if seeds != [ABLATION_BASELINE_SEED]:
        print(f"[WARN] ablation co nhieu seed: {seeds}")

    grid = defaultdict(dict)  # grid[backbone][experiment] = row
    for r in test:
        grid[r["backbone"]][r["experiment"]] = r
    order = order_backbones(grid)
    ablations = [a for a in ABLATION_ORDER if any(a in grid[b] for b in order)]

    run_ids = sorted({r["run_id"] for r in test})
    baseline_ids = sorted({r["baseline_run_id"] for r in test if r["baseline_run_id"]})

    width = max(len(BACKBONE_LABEL.get(b, b)) for b in order)
    body = []
    for b in order:
        label = BACKBONE_LABEL.get(b, b)
        first = next(iter(grid[b].values()))
        cells = [
            f"${float(first['main_seed42_accuracy']) * 100:.{precision}f}$",
            f"${float(first['main_seed42_f1_macro']) * 100:.{precision}f}$",
        ]
        for exp in ablations:
            r = grid[b].get(exp)
            if r is None:
                cells += ["--", "--"]
                continue
            cells.append(fmt_delta(float(r["accuracy_delta_vs_main"]), precision))
            cells.append(fmt_delta(float(r["f1_macro_delta_vs_main"]), precision))
        body.append(f"{label:<{width}} & " + " & ".join(cells) + r" \\")

    # header 2 tang
    group_head = [r"\multicolumn{2}{c}{\textbf{Main (seed 42)}}"]
    group_head += [
        rf"\multicolumn{{2}}{{c}}{{\textbf{{{ABLATION_LABEL.get(a, a)}}}}}"
        for a in ablations
    ]
    sub_head = [r"Acc.\ (\%)", r"F1 (\%)"] + [
        c for _ in ablations for c in (r"$\Delta$Acc.", r"$\Delta$F1")
    ]
    n_groups = 1 + len(ablations)
    cmid = " ".join(
        rf"\cmidrule(lr){{{2 * i + 2}-{2 * i + 3}}}" for i in range(n_groups)
    )

    caption = (
        "Effect of each ablation on Task B test performance, measured against the\n"
        "main configuration at the same seed and the same backbone. The two leftmost\n"
        "columns give the absolute main-run scores for seed 42; every $\\Delta$ column\n"
        "is the ablation minus that same-backbone main run, in percentage points.\n"
        "Each ablation was trained with a single seed (42) only, so no dispersion can\n"
        "be estimated for these runs and the differences below are single-run point\n"
        "estimates rather than averaged effects. For reference, the seed-to-seed\n"
        "standard deviation of the main runs in Table~\\ref{tab:task_b} is of comparable\n"
        "magnitude to most of these differences."
    )

    tex = "\n".join(
        [
            "% Table: Task B ablations vs main seed 42 (same backbone)",
            "% LUU Y: moi ablation chi co 1 seed (42) -> khong co +/- , chi la diem don.",
            *lineage_header(source, meta.get("registry_created_at_utc", "n/a"), run_ids),
            f"% Baseline run ids : {len(baseline_ids)} runs",
            *[f"%   {rid}" for rid in baseline_ids],
            r"\begin{table}[!htbp]",
            r"\centering",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{3pt}",
            r"\caption{" + caption + "}",
            r"\label{tab:ablation}",
            r"\begin{tabular}{l" + "cc" * n_groups + "}",
            r"\toprule",
            r"\textbf{Architecture} & " + " & ".join(group_head) + r" \\",
            cmid,
            r" & " + " & ".join(sub_head) + r" \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return tex, sorted(set(run_ids) | set(baseline_ids))


# --------------------------------------------------------------------------
# Bang 3 -- per-species F1 day du (mean +/- SD tren 3 main seed)
# --------------------------------------------------------------------------
def build_perspecies(rows, meta, source, precision, lang="en") -> tuple[str, list[str]]:
    test = [
        r
        for r in rows
        if r["split"] == "test" and r["experiment"] == "main"
    ]
    if not test:
        raise SystemExit("[ERROR] per_class_metrics khong co dong main/test")

    # f1[(species, backbone)] = [f1 tung seed]
    f1 = defaultdict(list)
    support = {}
    for r in test:
        f1[(r["class_name"], r["backbone"])].append(float(r["f1"]))
        support[r["class_name"]] = int(r["support"])

    seeds = verify_seeds({r["seed"] for r in test}, "per_class_metrics.csv")
    backbones = order_backbones({r["backbone"] for r in test})
    species = sorted(
        support,
        key=lambda s: (
            FAMILY_ORDER.index(SPECIES_FAMILY.get(s, "Cupressaceae"))
            if SPECIES_FAMILY.get(s) in FAMILY_ORDER
            else len(FAMILY_ORDER),
            s,
        ),
    )
    run_ids = sorted({r["run_id"] for r in test})

    # canh bao neu so seed khong dong deu
    counts = {len(v) for v in f1.values()}
    if counts != {len(MAIN_SEEDS)}:
        print(f"[WARN] so seed khong dong deu giua cac (loai, backbone): {counts}")

    body = []
    prev_family = None
    for sp in species:
        fam = SPECIES_FAMILY.get(sp, "n/a")
        if prev_family is not None and fam != prev_family:
            body.append(r"\midrule")
        if fam != prev_family:
            body.append(rf"\multicolumn{{{len(backbones) + 2}}}{{l}}{{\textit{{{fam}}}}} \\")
            prev_family = fam
        cells = []
        for b in backbones:
            vals = f1[(sp, b)]
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            cells.append(fmt_pm(statistics.fmean(vals), sd, precision))
        body.append(
            rf"\quad \textit{{{sp}}} & {support[sp]} & " + " & ".join(cells) + r" \\"
        )

    n_test = meta.get("split_counts", {}).get("test", sum(support.values()))
    caption = (
        f"Per-species F1 score (\\%) on the Task B test set ({len(species)} species,\n"
        f"{n_test} images), for every backbone. Values are the mean $\\pm$ standard\n"
        f"deviation across the {len(seeds)} main training seeds "
        f"({', '.join(seeds)}); the spread is a\n"
        f"standard deviation over three runs and is not an interval estimate.\n"
        f"Species are grouped by family. $n$ is the number of test images per species."
    )

    head = " & ".join(rf"\textbf{{{BACKBONE_LABEL.get(b, b)}}}" for b in backbones)
    tex = "\n".join(
        [
            "% Table: per-species F1, full 17 classes x 5 backbones",
            *lineage_header(source, meta.get("registry_created_at_utc", "n/a"), run_ids),
            r"\begin{table}[!htbp]",
            r"\centering",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{3pt}",
            r"\caption{" + caption + "}",
            r"\label{tab:perspecies_full}",
            # 7 cot voi ten loai dai vuot qua khoi chu 5.5in cua kieu preprint
            # mot cot; resizebox thu nho vua khit be ngang thay vi de tran le.
            r"\resizebox{\linewidth}{!}{%",
            r"\begin{tabular}{lr" + "c" * len(backbones) + "}",
            r"\toprule",
            r"\textbf{Species} & $n$ & " + head + r" \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table}",
            "",
        ]
    )
    return tex, run_ids


# --------------------------------------------------------------------------
# Bang 4 -- efficiency (params / MACs / model size)
# --------------------------------------------------------------------------
# LUU Y PROVENANCE: khac 3 bang tren, bang nay KHONG lay tu registry.
# params/MACs/model-size la thuoc tinh cua KIEN TRUC + kich thuoc anh vao,
# do bang scripts/profile_efficiency.py voi pretrained=False, nen khong phu
# thuoc checkpoint hay lan huan luyen nao. Cot latency bi BO vi do tren
# RTX 3060 trong khi moi run trong registry chay tren RTX 4090 -> khong so
# sanh duoc voi bat ky so nao khac trong bai.
EFFICIENCY_CSV = ROOT / "results/efficiency.csv"


def build_efficiency(rows, source: Path, precision: int = 2, lang="en") -> str:
    by = {r["model"]: r for r in rows}
    order = order_backbones(by)
    width = max(len(BACKBONE_LABEL.get(b, b)) for b in order)

    body = []
    for b in order:
        r = by[b]
        params = float(r["params_m"])
        # cot flops_g cua script dem FLOPs = 2 x MACs; bao cao MACs cho khop
        # voi quy uoc pho bien trong literature thi phai chia doi.
        macs = float(r["flops_g"]) / 2.0
        size = float(r["model_size_mb"])
        label = BACKBONE_LABEL.get(b, b)
        body.append(
            f"{label:<{width}} & ${params:.{precision}f}$ & ${macs:.{precision}f}$ "
            f"& ${size:.{precision}f}$ \\\\"
        )

    devices = sorted({r["device"] for r in rows})
    caption = (
        "Model complexity of the five architectures at the $224\\times224$ input\n"
        "resolution used throughout, with a 17-class head. Parameter counts, "
        "multiply--accumulate\noperations (MACs) and model size are properties of the "
        "architecture: they were\nmeasured on randomly initialised models and do not "
        "depend on any trained\ncheckpoint. MACs are reported rather than FLOPs; the "
        "profiling script records\n$\\text{FLOPs} = 2\\times\\text{MACs}$, and the values below are "
        "that column halved, so\nthat they are directly comparable with the convention "
        "used in the architecture\nliterature. Inference latency is not reported: it was "
        "measured on different\nhardware from the training and evaluation runs and is "
        "therefore not comparable\nwith any other number in this paper."
    )

    rel_source = source.relative_to(ROOT) if source.is_relative_to(ROOT) else source
    return "\n".join(
        [
            "% Table: model complexity (params / MACs / size)",
            "% AUTO-GENERATED bang scripts/person3/make_result_tables.py -- khong sua tay",
            f"% Source file      : {rel_source}",
            "% KHONG thuoc registry 20260819T043733Z: day la thuoc tinh kien truc,",
            "% do voi pretrained=False nen doc lap voi checkpoint va lan huan luyen.",
            f"% Latency da BO (do tren {', '.join(devices)}, khac phan cung cua registry).",
            "% Cot MACs = cot flops_g cua CSV chia 2 (CSV dem FLOPs = 2 x MACs).",
            r"\begin{table}[!htbp]",
            r"\centering",
            r"\caption{" + caption + "}",
            r"\label{tab:efficiency}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"\textbf{Architecture} & \textbf{Params (M)} & \textbf{MACs (G)} "
            r"& \textbf{Model size (MB)} \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


# --------------------------------------------------------------------------
def check_forbidden_language(name: str, text: str) -> None:
    low = text.lower()
    for bad in FORBIDDEN:
        if bad in low:
            raise SystemExit(f"[ERROR] {name} chua ngon ngu bi cam: {bad!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry-dir", type=Path, default=DEFAULT_REGISTRY)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--precision", type=int, default=3, help="so chu so thap phan (%%)")
    args = ap.parse_args()

    reg = args.registry_dir
    meta = read_metadata(reg)
    registry_ts = meta.get("registry_created_at_utc", "n/a")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ("tab_taskb.tex", "main_test_summary.csv", build_taskb),
        ("tab_ablation.tex", "ablation_vs_main_seed42.csv", build_ablation),
        ("tab_perspecies_full.tex", "per_class_metrics.csv", build_perspecies),
    ]

    lineage = {}
    for out_name, csv_name, builder in specs:
        source = reg / csv_name
        tex, run_ids = builder(read_csv(source), meta, source, args.precision)
        check_forbidden_language(out_name, tex)
        (args.out_dir / out_name).write_text(tex, encoding="utf-8")
        lineage[out_name] = {
            "source_file": rel(source),
            "registry_created_at_utc": registry_ts,
            "source_run_ids": run_ids,
            "n_source_runs": len(run_ids),
        }
        print(f"[OK] {out_name:<26} <- {csv_name} ({len(run_ids)} runs)")

    # bang efficiency: nguon khac, lineage khac -> xu ly rieng
    if EFFICIENCY_CSV.exists():
        eff_rows = read_csv(EFFICIENCY_CSV)
        eff_tex = build_efficiency(eff_rows, EFFICIENCY_CSV, precision=2)
        check_forbidden_language("tab_efficiency.tex", eff_tex)
        (args.out_dir / "tab_efficiency.tex").write_text(eff_tex, encoding="utf-8")
        lineage["tab_efficiency.tex"] = {
            "source_file": rel(EFFICIENCY_CSV),
            "registry_created_at_utc": None,
            "in_registry": False,
            "note": (
                "architecture-level complexity measured with pretrained=False; "
                "independent of any checkpoint. Latency column dropped: measured on "
                + ", ".join(sorted({r["device"] for r in eff_rows}))
                + ", not the registry hardware. MACs = csv flops_g / 2."
            ),
            "source_run_ids": [],
            "n_source_runs": 0,
        }
        print(f"[OK] {'tab_efficiency.tex':<26} <- {EFFICIENCY_CSV.name} (ngoai registry)")

    metadata = {
        "generated_by": "scripts/person3/make_result_tables.py",
        "registry_dir": rel(reg),
        "registry_created_at_utc": registry_ts,
        "split_version": meta.get("split_version"),
        "git_commit_of_registry": meta.get("git_commit"),
        "main_seeds": list(MAIN_SEEDS),
        "ablation_seed": ABLATION_BASELINE_SEED,
        "ablation_single_seed_only": True,
        "dispersion_reported_as": "standard deviation over 3 seeds (not an interval estimate)",
        "classes": meta.get("classes", []),
        "split_counts": meta.get("split_counts", {}),
        "tables": lineage,
        "all_source_run_ids": sorted(
            {rid for t in lineage.values() for rid in t["source_run_ids"]}
        ),
    }
    out_meta = args.out_dir / "tables_metadata.json"
    out_meta.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] {'tables_metadata.json':<26} ({len(metadata['all_source_run_ids'])} runs tong)")


if __name__ == "__main__":
    main()
