"""Build Phase 2 learning-curve and efficiency artifacts for WoodID.

This module is invoked by ``01_learning_curves_and_efficiency.ipynb``.  It is
kept beside the notebook so the extraction and profiling logic is reviewable,
testable, and reusable without changing the Phase 1 training code.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/woodid-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.models import create_model


BACKBONES = [
    "resnet50",
    "efficientnet_b3",
    "mobilenetv3_large",
    "convnext_tiny",
    "swin_tiny",
]
DISPLAY_NAMES = {
    "resnet50": "ResNet-50",
    "efficientnet_b3": "EfficientNet-B3",
    "mobilenetv3_large": "MobileNetV3-Large",
    "convnext_tiny": "ConvNeXt-Tiny",
    "swin_tiny": "Swin-Tiny",
}
FAMILY_BY_CLASS = {
    "Araucaria araucana": "Araucariaceae",
    "Callitris columellaris": "Cupressaceae",
    "Calocedrus rupestris": "Cupressaceae",
    "Calocedrus sp": "Cupressaceae",
    "Chamaecyparis obtusa": "Cupressaceae",
    "Cryptomeria japonica": "Cupressaceae",
    "Cunninghamia lanceolata": "Cupressaceae",
    "Cupressus funebris (Cupressus pendula)": "Cupressaceae",
    "Fokienia hodginsii": "Cupressaceae",
    "Thujopsis dolabrata": "Cupressaceae",
    "Xanthocyparis vietnamensis": "Cupressaceae",
    "Picea glauca": "Pinaceae",
    "Pinus cernua": "Pinaceae",
    "Pinus dalatensis": "Pinaceae",
    "Pinus krempfii": "Pinaceae",
    "Pinus radiata": "Pinaceae",
    "Pinus sp": "Pinaceae",
    "Pseudotsuga brevifolia": "Pinaceae",
    "Podocarpus pilgeri": "Podocarpaceae",
    "Taxus wallichiana": "Taxaceae",
}
EXCLUDED_CLASSES = {
    "Calocedrus sp",
    "Cupressus funebris (Cupressus pendula)",
    "Pinus sp",
}
TRAIN_NOTEBOOKS = {
    "resnet50": "01_train_resnet50.ipynb",
    "efficientnet_b3": "02_train_efficientnet_b3.ipynb",
    "mobilenetv3_large": "03_train_mobilenetv3_large.ipynb",
    "convnext_tiny": "04_train_convnext_tiny.ipynb",
    "swin_tiny": "05_train_swin_tiny.ipynb",
}
SEEDS = (42, 123, 456)
LEARNING_SEEDS = (42,)

RUN_START_RE = re.compile(
    r"^\[main \| (?P<backbone>[a-z0-9_]+) s(?P<seed>42|123|456)\] "
    r"train=3365, val=903, test=532, classes=17$"
)
RUN_END_RE = re.compile(
    r"^\[main \| (?P<backbone>[a-z0-9_]+) s(?P<seed>42|123|456)\] "
    r"val_acc="
)
EPOCH_RE = re.compile(
    r"^Epoch (?P<epoch>\d+)/(?P<budget>\d+) \| "
    r"Train Loss: (?P<train_loss>[0-9.]+), Acc: (?P<train_acc>[0-9.]+)% \| "
    r"Val Loss: (?P<val_loss>[0-9.]+), Acc: (?P<val_acc>[0-9.]+)% \| "
    r"LR: (?P<lr>[0-9.eE+-]+)"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_paths(root: Path) -> dict[str, Path]:
    root = Path(root).resolve()
    paper_root = root.parent / "WoodID"
    paths = {
        "root": root,
        "paper_root": paper_root,
        "phase1": root / "notebooks" / "phase 1" / "task-B-17-new",
        "registry": root
        / "docs"
        / "reports"
        / "experiments"
        / "phase_1b"
        / "final_b_new_recomputed"
        / "20260819T043733Z"
        / "task-b-17class-20260819T043733Z-registry.csv",
        "registry_metadata": root
        / "docs"
        / "reports"
        / "experiments"
        / "phase_1b"
        / "final_b_new_recomputed"
        / "20260819T043733Z"
        / "metadata.json",
        "main_summary": root
        / "docs"
        / "reports"
        / "experiments"
        / "phase_1b"
        / "final_b_new_recomputed"
        / "20260819T043733Z"
        / "main_test_summary.csv",
        "ablation_summary": root
        / "docs"
        / "reports"
        / "experiments"
        / "phase_1b"
        / "final_b_new_recomputed"
        / "20260819T043733Z"
        / "ablation_vs_main_seed42.csv",
        "per_class_metrics": root
        / "docs"
        / "reports"
        / "experiments"
        / "phase_1b"
        / "final_b_new_recomputed"
        / "20260819T043733Z"
        / "per_class_metrics.csv",
        "split_manifest": root
        / "docs"
        / "reports"
        / "phase_1"
        / "phase_1_leakage_origin_investigation"
        / "task-B-clean_group_split_plan.csv",
        "experiment": root
        / "data"
        / "experiments"
        / "Task-B-new"
        / "19-08-2026"
        / "main",
        "learning_out": paper_root / "artifact" / "person2" / "learning_curves",
        "efficiency_out": paper_root / "artifact" / "person2" / "efficiency",
        "methods_out": paper_root / "artifact" / "person2" / "methods",
    }
    for required in (
        "paper_root",
        "phase1",
        "registry",
        "main_summary",
        "ablation_summary",
        "per_class_metrics",
        "split_manifest",
        "experiment",
    ):
        if not paths[required].exists():
            raise FileNotFoundError(f"Missing required path: {paths[required]}")
    paths["learning_out"].mkdir(parents=True, exist_ok=True)
    paths["efficiency_out"].mkdir(parents=True, exist_ok=True)
    paths["methods_out"].mkdir(parents=True, exist_ok=True)
    return paths


def _scientific_name_tex(name: str) -> str:
    if name.endswith(" sp"):
        return rf"\textit{{{name.removesuffix(' sp')}}} sp."
    if name == "Cupressus funebris (Cupressus pendula)":
        return r"\textit{Cupressus funebris} (\textit{C. pendula})"
    return rf"\textit{{{name}}}"


def build_methods_artifacts(root: Path) -> dict[str, Any]:
    """Generate dataset/augmentation tables from the immutable local sources."""
    paths = repo_paths(root)
    manifest = pd.read_csv(paths["split_manifest"])
    if set(manifest["class_name"].unique()) != set(FAMILY_BY_CLASS):
        raise AssertionError("Taxa in split manifest do not match the reviewed taxonomy map")
    if len(manifest) != 5572 or manifest["origin_key"].nunique() != 557:
        raise AssertionError("Clean collection totals differ from 5,572 images / 557 origin keys")

    manifest = manifest.assign(family=manifest["class_name"].map(FAMILY_BY_CLASS))
    benchmark = manifest[~manifest["class_name"].isin(EXCLUDED_CLASSES)].copy()
    if len(benchmark) != 4800 or benchmark["origin_key"].nunique() != 552:
        raise AssertionError("Benchmark totals differ from 4,800 images / 552 origin keys")
    split_counts = benchmark["new_split"].value_counts().to_dict()
    split_origins = benchmark.groupby("new_split")["origin_key"].nunique().to_dict()
    if split_counts != {"train": 3365, "val": 903, "test": 532}:
        raise AssertionError(f"Unexpected benchmark image split: {split_counts}")
    if split_origins != {"test": 76, "train": 386, "val": 90}:
        raise AssertionError(f"Unexpected benchmark origin split: {split_origins}")
    origin_splits = benchmark.groupby("origin_key")["new_split"].nunique()
    if not origin_splits.eq(1).all():
        raise AssertionError("An origin key occurs in more than one final split")

    out = paths["methods_out"]
    family_full = manifest.groupby("family").agg(
        taxa=("class_name", "nunique"), images=("class_name", "size"), origins=("origin_key", "nunique")
    )
    family_bench = benchmark.groupby("family").agg(
        taxa=("class_name", "nunique"), images=("class_name", "size"), origins=("origin_key", "nunique")
    )
    family_lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Clean collection} & \multicolumn{3}{c}{17-species benchmark} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Family & Taxa & Images & Origins & Species & Images & Origins \\",
        r"\midrule",
    ]
    for family in sorted(set(FAMILY_BY_CLASS.values())):
        full = family_full.loc[family]
        selected = family_bench.loc[family]
        family_lines.append(
            f"{family} & {full.taxa} & {full.images:,} & {full.origins} & "
            f"{selected.taxa} & {selected.images:,} & {selected.origins} \\\\"
        )
    family_lines.extend(
        [
            r"\midrule",
            r"Total & 20 & 5,572 & 557 & 17 & 4,800 & 552 \\",
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    (out / "tab_species.tex").write_text("\n".join(family_lines) + "\n", encoding="utf-8")

    taxa = manifest.groupby(["family", "class_name"], as_index=False).agg(
        images=("class_name", "size"), origins=("origin_key", "nunique")
    )
    full_lines = [
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llrrc}",
        r"\toprule",
        r"Family & Taxon label & Images & Origin keys & Included \\",
        r"\midrule",
    ]
    previous_family = None
    for row in taxa.sort_values(["family", "class_name"]).itertuples(index=False):
        if previous_family is not None and row.family != previous_family:
            full_lines.append(r"\midrule")
        included = "No" if row.class_name in EXCLUDED_CLASSES else "Yes"
        full_lines.append(
            f"{row.family} & {_scientific_name_tex(row.class_name)} & "
            f"{row.images:,} & {row.origins} & {included} \\\\"
        )
        previous_family = row.family
    full_lines.extend([r"\bottomrule", r"\end{tabular}}"])
    (out / "tab_species_full.tex").write_text("\n".join(full_lines) + "\n", encoding="utf-8")

    augmentation_lines = [
        r"\begin{tabular}{p{0.23\linewidth}p{0.69\linewidth}}",
        r"\toprule",
        r"Configuration & Training transform pipeline \\",
        r"\midrule",
        r"Main & Resize to $224\times224$; horizontal flip ($p=0.5$); vertical flip ($p=0.5$); random rotation within $\pm30^\circ$; brightness and contrast jitter (amplitude 0.2); ImageNet normalization. \\",
        r"Discrete rotation ablation & Same pipeline, except the continuous rotation is replaced by a uniform choice from $\{0^\circ,90^\circ,180^\circ,270^\circ\}$. \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (out / "tab_augconfig.tex").write_text(
        "\n".join(augmentation_lines) + "\n", encoding="utf-8"
    )

    test_origins = benchmark[benchmark["new_split"].eq("test")].groupby("class_name")["origin_key"].nunique()
    run_configs = sorted((paths["experiment"] / "seed_42").glob("*/run_config.json"))
    if len(run_configs) != len(BACKBONES):
        raise AssertionError(f"Expected five seed-42 main run configs, found {len(run_configs)}")
    metadata = {
        "created_utc": utc_now(),
        "split_manifest": str(paths["split_manifest"].relative_to(root)),
        "split_manifest_sha256": sha256(paths["split_manifest"]),
        "assignment_column": "new_split",
        "taxonomy_mapping": FAMILY_BY_CLASS,
        "excluded_classes": sorted(EXCLUDED_CLASSES),
        "verified": {
            "clean_collection": {"taxa": 20, "images": 5572, "origin_keys": 557, "families": 5},
            "benchmark": {"species": 17, "images": 4800, "origin_keys": 552, "families": 5},
            "split_images": split_counts,
            "split_origin_keys": split_origins,
            "cross_split_origin_key_overlap": 0,
            "test_origins_per_species": {
                "mean": float(test_origins.mean()),
                "median": float(test_origins.median()),
                "min": int(test_origins.min()),
                "max": int(test_origins.max()),
                "species_with_one": int(test_origins.eq(1).sum()),
            },
        },
        "augmentation_source": "src/data/transforms.py and registry augmentation field",
        "augmentation_source_sha256": sha256(root / "src" / "data" / "transforms.py"),
        "registry_metadata": str(paths["registry_metadata"].relative_to(root)),
        "registry_metadata_sha256": sha256(paths["registry_metadata"]),
        "seed42_main_run_configs": [
            {"path": str(path.relative_to(root)), "sha256": sha256(path)}
            for path in run_configs
        ],
    }
    (out / "methods_sources.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metadata


def notebook_output_lines(path: Path) -> list[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    lines: list[str] = []
    for cell in notebook.get("cells", []):
        for output in cell.get("outputs", []):
            text = output.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            lines.extend(str(text).splitlines())
    return lines


def parse_main_segments(path: Path) -> dict[tuple[str, int], list[list[dict[str, Any]]]]:
    """Parse literal Phase 1 stdout; repeated epoch 1 starts a new attempt."""
    segments: dict[tuple[str, int], list[list[dict[str, Any]]]] = {}
    active_key: tuple[str, int] | None = None
    active_rows: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal active_rows
        if active_key is not None and active_rows:
            segments.setdefault(active_key, []).append(active_rows)
        active_rows = []

    for line in notebook_output_lines(path):
        start = RUN_START_RE.match(line)
        if start:
            flush()
            active_key = (start.group("backbone"), int(start.group("seed")))
            continue
        if active_key is None:
            continue
        epoch = EPOCH_RE.match(line)
        if epoch:
            row = {
                "epoch": int(epoch.group("epoch")),
                "epoch_budget": int(epoch.group("budget")),
                "train_loss": float(epoch.group("train_loss")),
                "train_acc": float(epoch.group("train_acc")),
                "val_loss": float(epoch.group("val_loss")),
                "val_acc": float(epoch.group("val_acc")),
                "lr": float(epoch.group("lr")),
            }
            if row["epoch"] == 1 and active_rows:
                flush()
            active_rows.append(row)
            continue
        if RUN_END_RE.match(line):
            flush()
            active_key = None
    flush()
    return segments


def select_matching_segment(
    candidates: list[list[dict[str, Any]]], metrics: dict[str, Any]
) -> list[dict[str, Any]]:
    """Select the stdout attempt that matches the committed run summary."""
    epochs_trained = int(metrics["epochs_trained"])
    length_matches = [segment for segment in candidates if len(segment) == epochs_trained]
    if not length_matches:
        raise ValueError(
            f"No stdout segment has epochs_trained={epochs_trained}; "
            f"candidate lengths={[len(item) for item in candidates]}"
        )

    def score(segment: list[dict[str, Any]]) -> float:
        return (
            abs(max(row["val_acc"] for row in segment) - float(metrics["best_val_acc"]))
            + abs(segment[-1]["train_acc"] - float(metrics["final_train_acc"]))
            + abs(segment[-1]["val_acc"] - float(metrics["final_val_acc"]))
        )

    selected = min(length_matches, key=score)
    if score(selected) > 0.08:
        raise ValueError(f"Selected stdout segment does not match run summary; score={score(selected):.4f}")
    return selected


def build_learning_curves(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    paths = repo_paths(root)
    registry = pd.read_csv(paths["registry"])
    registry = registry[registry["experiment"].eq("main")].copy()
    all_rows: list[dict[str, Any]] = []
    source_notebooks: list[dict[str, Any]] = []

    for backbone in BACKBONES:
        notebook_path = paths["phase1"] / TRAIN_NOTEBOOKS[backbone]
        parsed = parse_main_segments(notebook_path)
        source_notebooks.append(
            {
                "path": str(notebook_path.relative_to(paths["root"])),
                "sha256": sha256(notebook_path),
            }
        )
        for seed in LEARNING_SEEDS:
            key = (backbone, seed)
            run = registry[(registry["backbone"] == backbone) & (registry["seed"] == seed)]
            if len(run) != 1:
                raise ValueError(f"Expected one registry row for {key}, found {len(run)}")
            run_row = run.iloc[0]
            metrics_path = paths["experiment"] / f"seed_{seed}" / backbone / "test_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            selected = select_matching_segment(parsed.get(key, []), metrics)
            for row in selected:
                all_rows.append(
                    {
                        "run_id": run_row["run_id"],
                        "backbone": backbone,
                        "display_name": DISPLAY_NAMES[backbone],
                        "seed": seed,
                        "experiment": "main",
                        **row,
                        "source_notebook": str(notebook_path.relative_to(paths["root"])),
                        "source_metrics": str(metrics_path.relative_to(paths["root"])),
                    }
                )

    curves = pd.DataFrame(all_rows).sort_values(["backbone", "seed", "epoch"])
    expected = {(backbone, seed) for backbone in BACKBONES for seed in LEARNING_SEEDS}
    actual = set(zip(curves["backbone"], curves["seed"]))
    if actual != expected:
        raise AssertionError(f"Learning-curve runs mismatch: missing={expected - actual}")

    summary = (
        curves.groupby(["backbone", "display_name", "epoch"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            train_loss_mean=("train_loss", "mean"),
            train_loss_sd=("train_loss", "std"),
            train_acc_mean=("train_acc", "mean"),
            train_acc_sd=("train_acc", "std"),
            val_loss_mean=("val_loss", "mean"),
            val_loss_sd=("val_loss", "std"),
            val_acc_mean=("val_acc", "mean"),
            val_acc_sd=("val_acc", "std"),
        )
        .sort_values(["backbone", "epoch"])
    )

    out = paths["learning_out"]
    curves.to_csv(out / "learning_curves.csv", index=False)
    summary.to_csv(out / "learning_curves_summary.csv", index=False)
    plot_learning_curves(curves, out)

    metadata = {
        "created_utc": utc_now(),
        "scope": "Task B, 17 classes, main benchmark, seed 42 for all five architectures",
        "method": "Parsed literal per-epoch stdout from committed Phase 1 training notebooks",
        "no_interpolation": True,
        "registry": str(paths["registry"].relative_to(paths["root"])),
        "registry_sha256": sha256(paths["registry"]),
        "source_run_ids": sorted(curves["run_id"].unique().tolist()),
        "source_notebooks": source_notebooks,
        "validation": {
            "runs": int(curves[["backbone", "seed"]].drop_duplicates().shape[0]),
            "rows": int(len(curves)),
            "selection": "stdout segment must match epochs_trained and final/best metrics JSON",
            "seed_choice": "Seed 42 is used consistently because no stored stdout segment matches the final ResNet-50 seed-456 checkpoint summary.",
            "excluded_history": "ResNet-50 seed 456 contains repeated attempts after a no-space-left-on-device failure; none matches its final test_metrics.json, so three-seed curves would mix lineages.",
        },
        "training_hardware_evidence": {
            "device": "NVIDIA GeForce RTX 4090",
            "source": "stdout stored in each source training notebook",
        },
    }
    (out / "learning_curves_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return curves, summary, metadata


def plot_learning_curves(curves: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(len(BACKBONES), 2, figsize=(12, 16), sharex=False)
    colors = {"train": "#2878B5", "validation": "#D95319"}
    for row_index, backbone in enumerate(BACKBONES):
        subset = curves[curves["backbone"] == backbone]
        common_epoch = int(subset["epoch"].max())
        common = subset
        aggregate = common.groupby("epoch", as_index=False).agg(
            train_loss_mean=("train_loss", "mean"),
            train_loss_sd=("train_loss", "std"),
            val_loss_mean=("val_loss", "mean"),
            val_loss_sd=("val_loss", "std"),
            train_acc_mean=("train_acc", "mean"),
            train_acc_sd=("train_acc", "std"),
            val_acc_mean=("val_acc", "mean"),
            val_acc_sd=("val_acc", "std"),
        )
        for metric, label, color in (
            ("train_loss", "Train", colors["train"]),
            ("val_loss", "Validation", colors["validation"]),
        ):
            mean = aggregate[f"{metric}_mean"].to_numpy()
            sd = aggregate[f"{metric}_sd"].fillna(0).to_numpy()
            epoch = aggregate["epoch"].to_numpy()
            axes[row_index, 0].plot(epoch, mean, color=color, label=label)
        for metric, label, color in (
            ("train_acc", "Train", colors["train"]),
            ("val_acc", "Validation", colors["validation"]),
        ):
            mean = aggregate[f"{metric}_mean"].to_numpy()
            sd = aggregate[f"{metric}_sd"].fillna(0).to_numpy()
            epoch = aggregate["epoch"].to_numpy()
            axes[row_index, 1].plot(epoch, mean, color=color, label=label)
            # No uncertainty band: these are literal seed-42 histories.
        best = subset.loc[subset["val_acc"].idxmax()]
        axes[row_index, 1].scatter(
            [best["epoch"]],
            [best["val_acc"]],
            marker="*",
            s=80,
            color="#222222",
            label=f"Best val (epoch {int(best['epoch'])})",
            zorder=5,
        )
        axes[row_index, 0].set_ylabel(f"{DISPLAY_NAMES[backbone]}\nLoss")
        axes[row_index, 1].set_ylabel("Accuracy (%)")
        axes[row_index, 0].set_title(f"Loss (seed 42; epochs 1–{common_epoch})")
        axes[row_index, 1].set_title(f"Accuracy (seed 42; epochs 1–{common_epoch})")
        for axis in axes[row_index]:
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.25)
            axis.legend(loc="best")
    fig.suptitle("Task B (17 classes): main-run learning curves for seed 42", y=1.002)
    fig.tight_layout()
    fig.savefig(out / "fig_learning_curves.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_learning_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def command_output(args: list[str]) -> str | None:
    try:
        return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def live_environment() -> dict[str, Any]:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    properties = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cpu": platform.processor() or command_output(["lscpu"]),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": str(torch.backends.cudnn.version()),
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu_name,
        "gpu_memory_mib": int(properties.total_memory / 1024**2) if properties else None,
        "nvidia_smi": command_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
        ),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("timm", "thop", "numpy", "pandas")
        },
    }


def resolve_checkpoint(paths: dict[str, Path], backbone: str) -> Path:
    folder = paths["experiment"] / "seed_42" / backbone
    matches = sorted(folder.glob(f"{backbone}_seed42_best.pth"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one seed-42 best checkpoint in {folder}, found {matches}")
    return matches[0]


def state_dict_from_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, dict):
        raise TypeError(f"No state dict in {path}")
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def measure_latency(
    model: torch.nn.Module,
    dummy: torch.Tensor,
    warmup: int,
    repetitions: int,
) -> tuple[np.ndarray, float, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            model(dummy)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline_mib = torch.cuda.memory_allocated() / 1024**2
        samples: list[float] = []
        for _ in range(repetitions):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(dummy)
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)))
        peak_mib = torch.cuda.max_memory_allocated() / 1024**2
    return np.asarray(samples, dtype=float), baseline_mib, peak_mib


def profile_checkpoint(
    root: Path,
    paths: dict[str, Path],
    registry_row: pd.Series,
    warmup: int,
    repetitions: int,
) -> dict[str, Any]:
    backbone = str(registry_row["backbone"])
    checkpoint_path = resolve_checkpoint(paths, backbone)
    state = state_dict_from_checkpoint(checkpoint_path)
    model = create_model(backbone, num_classes=17, pretrained=False)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"State-dict mismatch for {backbone}: {incompatible}")

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    state_dict_bytes = sum(tensor.numel() * tensor.element_size() for tensor in state.values())
    model = model.cuda().eval()
    dummy = torch.randn(1, 3, 224, 224, device="cuda")

    from thop import profile as thop_profile

    macs, thop_params = thop_profile(model, inputs=(dummy,), verbose=False)
    samples, baseline_mib, peak_mib = measure_latency(model, dummy, warmup, repetitions)
    row = {
        "profile_id": f"efficiency__seed_42__{backbone}",
        "source_run_id": str(registry_row["run_id"]),
        "backbone": backbone,
        "display_name": DISPLAY_NAMES[backbone],
        "seed": 42,
        "num_classes": 17,
        "image_size": 224,
        "batch_size": 1,
        "precision": "FP32",
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "thop_params": int(thop_params),
        "gmacs": float(macs / 1e9),
        "gflops_assuming_2_per_mac": float(2 * macs / 1e9),
        "state_dict_size_mib": float(state_dict_bytes / 1024**2),
        "checkpoint_file_size_mib": float(checkpoint_path.stat().st_size / 1024**2),
        "latency_mean_ms": float(samples.mean()),
        "latency_median_ms": float(np.median(samples)),
        "latency_sd_ms": float(samples.std(ddof=1)),
        "latency_p95_ms": float(np.percentile(samples, 95)),
        "latency_min_ms": float(samples.min()),
        "latency_max_ms": float(samples.max()),
        "warmup_runs": warmup,
        "timed_runs": repetitions,
        "baseline_allocated_vram_mib": float(baseline_mib),
        "peak_allocated_vram_mib": float(peak_mib),
        "incremental_peak_vram_mib": float(peak_mib - baseline_mib),
        "test_accuracy": float(registry_row["test_accuracy"]),
        "test_f1_macro": float(registry_row["test_f1_macro"]),
        "checkpoint_path": str(checkpoint_path.relative_to(root)),
        "latency_samples_ms": samples.tolist(),
    }
    del model, dummy, state
    torch.cuda.empty_cache()
    return row


def latex_efficiency_table(results: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Architecture & Params (M) & GMACs & Size (MiB) & Latency (ms) & VRAM (MiB) & Seed-42 F1 (\%) \\",
        r"\midrule",
    ]
    for row in results.itertuples(index=False):
        lines.append(
            f"{row.display_name} & {row.total_params / 1e6:.2f} & {row.gmacs:.2f} & "
            f"{row.state_dict_size_mib:.1f} & {row.latency_mean_ms:.2f} $\\pm$ {row.latency_sd_ms:.2f} & "
            f"{row.peak_allocated_vram_mib:.0f} & {100 * row.test_f1_macro:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def latex_environment_table(metadata: dict[str, Any], path: Path) -> None:
    environments = metadata["environments"]
    training = environments["training"]
    evaluation = environments["official_metric_evaluation"]
    efficiency = environments["efficiency_profile"]
    lines = [
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"Stage & GPU & Python & PyTorch \\",
        r"\midrule",
        f"Training & {training['device']} & {training['python']} & -- \\\\",
        f"Official metric evaluation & {evaluation['device']} & {evaluation['python']} & {evaluation['pytorch']} \\\\",
        f"Efficiency profile & {efficiency['gpu']} & {efficiency['python']} & {efficiency['pytorch']} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_efficiency_radar(results: pd.DataFrame, out: Path) -> None:
    metrics = {
        "Accuracy": results["test_accuracy"].to_numpy(),
        "Macro F1": results["test_f1_macro"].to_numpy(),
        "Speed": 1 / results["latency_mean_ms"].to_numpy(),
        "Compactness": 1 / results["state_dict_size_mib"].to_numpy(),
        "Compute": 1 / results["gmacs"].to_numpy(),
    }
    normalized: dict[str, np.ndarray] = {}
    for name, values in metrics.items():
        low, high = float(values.min()), float(values.max())
        normalized[name] = np.ones_like(values) if high == low else (values - low) / (high - low)
    labels = list(normalized)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, axis = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    for index, row in results.reset_index(drop=True).iterrows():
        values = [normalized[label][index] for label in labels]
        values += values[:1]
        axis.plot(angles, values, linewidth=1.7, label=row["display_name"])
        axis.fill(angles, values, alpha=0.05)
    axis.set_xticks(angles[:-1], labels)
    axis.set_ylim(0, 1)
    axis.set_title("Relative architecture trade-offs (min–max normalized)", pad=24)
    axis.legend(loc="upper right", bbox_to_anchor=(1.38, 1.12))
    fig.tight_layout()
    fig.savefig(out / "fig_efficiency_radar.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_efficiency_radar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_efficiency_profile(
    root: Path, warmup: int = 50, repetitions: int = 200
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Efficiency artifacts were not generated; run this cell on the local RTX 3060."
        )
    paths = repo_paths(root)
    environment = live_environment()
    if environment["gpu"] != "NVIDIA GeForce RTX 3060":
        raise RuntimeError(f"Expected local RTX 3060, detected {environment['gpu']!r}")

    registry = pd.read_csv(paths["registry"])
    seed42 = registry[(registry["experiment"] == "main") & (registry["seed"] == 42)]
    rows = []
    for backbone in BACKBONES:
        match = seed42[seed42["backbone"] == backbone]
        if len(match) != 1:
            raise ValueError(f"Expected one seed-42 registry row for {backbone}, found {len(match)}")
        print(f"Profiling {DISPLAY_NAMES[backbone]} on {environment['gpu']} ...")
        rows.append(profile_checkpoint(root, paths, match.iloc[0], warmup, repetitions))

    sample_rows = []
    for row in rows:
        samples = row.pop("latency_samples_ms")
        sample_rows.extend(
            {
                "profile_id": row["profile_id"],
                "source_run_id": row["source_run_id"],
                "backbone": row["backbone"],
                "repetition": index + 1,
                "latency_ms": value,
            }
            for index, value in enumerate(samples)
        )
    results = pd.DataFrame(rows)
    out = paths["efficiency_out"]
    results.to_csv(out / "efficiency_17class.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(out / "latency_samples.csv", index=False)
    latex_efficiency_table(results, out / "tab_efficiency.tex")
    plot_efficiency_radar(results, out)
    registry_metadata = json.loads(paths["registry_metadata"].read_text(encoding="utf-8"))
    metadata = {
        "created_utc": utc_now(),
        "scope": "Task B, 17 classes, main seed-42 checkpoints",
        "protocol": {
            "input": [1, 3, 224, 224],
            "precision": "FP32",
            "warmup_runs": warmup,
            "timed_runs": repetitions,
            "latency_timer": "torch.cuda.Event; one device synchronization per sample",
            "macs": "THOP multiply-accumulate count",
            "flops": "reported as 2 FLOPs per MAC only in explicitly named column",
            "size": "raw tensor bytes in model state_dict; MiB = bytes / 1024^2",
            "vram": "torch.cuda.max_memory_allocated, includes model/input/forward allocations",
        },
        "environments": {
            "training": {
                "device": "NVIDIA GeForce RTX 4090",
                "python": "3.12.13",
                "source": "stored stdout in Phase 1 training notebooks",
            },
            "official_metric_evaluation": {
                "device": registry_metadata.get("device_name"),
                "python": registry_metadata.get("python_version"),
                "pytorch": registry_metadata.get("torch_version"),
                "source": str(paths["registry_metadata"].relative_to(root)),
            },
            "efficiency_profile": environment,
        },
        "registry": str(paths["registry"].relative_to(root)),
        "registry_sha256": sha256(paths["registry"]),
        "source_run_ids": results["source_run_id"].tolist(),
        "woodbench_git_commit": command_output(["git", "-C", str(root), "rev-parse", "HEAD"]),
        "important": "RTX 3060 latency is an independent deployment profile; accuracy/F1 come from the official RTX 4090 evaluation registry.",
    }
    latex_environment_table(metadata, out / "tab_environment.tex")
    (out / "efficiency_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return results, metadata


def _table_row(text: str, marker: str) -> str:
    matches = [line for line in text.splitlines() if marker in line and "&" in line]
    if len(matches) != 1:
        raise AssertionError(f"Expected one table row containing {marker!r}, found {len(matches)}")
    return matches[0]


def _assert_tokens_in_order(line: str, tokens: list[str], label: str) -> None:
    cursor = 0
    for token in tokens:
        position = line.find(token, cursor)
        if position < 0:
            raise AssertionError(f"{label}: missing or out-of-order value {token!r} in {line!r}")
        cursor = position + len(token)


def validate_integrated_result_tables(root: Path) -> dict[str, Any]:
    """Verify the paper's Task-B tables against the official 19-Aug registry."""
    paths = repo_paths(root)
    paper_results = paths["paper_root"] / "results"
    main_tex = (paper_results / "tab_taskb.tex").read_text(encoding="utf-8")
    ablation_tex = (paper_results / "tab_ablation.tex").read_text(encoding="utf-8")
    per_class_tex = (paper_results / "tab_perspecies_full.tex").read_text(encoding="utf-8")

    main = pd.read_csv(paths["main_summary"]).set_index("backbone")
    for backbone in BACKBONES:
        row = main.loc[backbone]
        line = _table_row(main_tex, DISPLAY_NAMES[backbone])
        _assert_tokens_in_order(
            line,
            [
                f"{100 * row.accuracy_mean:.3f} \\pm {100 * row.accuracy_std:.3f}",
                f"{100 * row.f1_macro_mean:.3f} \\pm {100 * row.f1_macro_std:.3f}",
            ],
            f"main/{backbone}",
        )

    ablation = pd.read_csv(paths["ablation_summary"])
    experiments = (
        "taskB_90rotation",
        "taskB_label_smoothing_0.0",
        "taskB_label_smoothing_0.2",
    )
    for backbone in BACKBONES:
        subset = ablation[ablation["backbone"].eq(backbone)].set_index("experiment")
        if set(subset.index) != set(experiments):
            raise AssertionError(f"Unexpected ablation rows for {backbone}: {sorted(subset.index)}")
        baseline = subset.loc[experiments[0]]
        expected = [
            f"{100 * baseline.main_seed42_accuracy:.3f}",
            f"{100 * baseline.main_seed42_f1_macro:.3f}",
        ]
        for experiment in experiments:
            row = subset.loc[experiment]
            expected.extend(
                [
                    f"{100 * row.accuracy_delta_vs_main:+.3f}",
                    f"{100 * row.f1_macro_delta_vs_main:+.3f}",
                ]
            )
        _assert_tokens_in_order(
            _table_row(ablation_tex, DISPLAY_NAMES[backbone]),
            expected,
            f"ablation/{backbone}",
        )

    per_class = pd.read_csv(paths["per_class_metrics"])
    per_class = per_class[
        per_class["experiment"].eq("main") & per_class["split"].eq("test")
    ]
    grouped = (
        per_class.groupby(["class_name", "backbone"], as_index=False)
        .agg(f1_mean=("f1", "mean"), f1_sd=("f1", "std"), support=("support", "first"))
    )
    species = per_class.sort_values("class_index")["class_name"].drop_duplicates().tolist()
    for class_name in species:
        subset = grouped[grouped["class_name"].eq(class_name)].set_index("backbone")
        supports = subset["support"].unique()
        if len(supports) != 1:
            raise AssertionError(f"Inconsistent test support for {class_name}: {supports.tolist()}")
        expected = [f"{int(supports[0])}"]
        expected.extend(
            f"{100 * subset.loc[backbone, 'f1_mean']:.3f} \\pm "
            f"{100 * subset.loc[backbone, 'f1_sd']:.3f}"
            for backbone in BACKBONES
        )
        _assert_tokens_in_order(
            _table_row(per_class_tex, rf"\textit{{{class_name}}}"),
            expected,
            f"per-class/{class_name}",
        )

    return {
        "registry": str(paths["registry"].relative_to(root)),
        "main_rows": len(BACKBONES),
        "ablation_rows": len(BACKBONES),
        "per_class_rows": len(species),
        "status": "all integrated numeric cells match the official 19-Aug registry",
    }


def validate_outputs(root: Path) -> dict[str, Any]:
    paths = repo_paths(root)
    required = [
        paths["methods_out"] / "tab_species.tex",
        paths["methods_out"] / "tab_species_full.tex",
        paths["methods_out"] / "tab_augconfig.tex",
        paths["methods_out"] / "methods_sources.json",
        paths["learning_out"] / "learning_curves.csv",
        paths["learning_out"] / "learning_curves_summary.csv",
        paths["learning_out"] / "learning_curves_metadata.json",
        paths["learning_out"] / "fig_learning_curves.pdf",
        paths["learning_out"] / "fig_learning_curves.png",
        paths["efficiency_out"] / "efficiency_17class.csv",
        paths["efficiency_out"] / "efficiency_metadata.json",
        paths["efficiency_out"] / "latency_samples.csv",
        paths["efficiency_out"] / "tab_efficiency.tex",
        paths["efficiency_out"] / "tab_environment.tex",
        paths["efficiency_out"] / "fig_efficiency_radar.pdf",
        paths["efficiency_out"] / "fig_efficiency_radar.png",
    ]
    missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise AssertionError(f"Missing/empty Phase 2 outputs: {missing}")
    return {
        "validated_utc": utc_now(),
        "files": [str(path) for path in required],
        "integrated_tables": validate_integrated_result_tables(root),
    }
