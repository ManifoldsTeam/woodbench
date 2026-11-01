#!/usr/bin/env python3
"""Step 1 report: one row per run and the chosen backbone LR per model.

Rule, fixed before looking at results: compare val macro-F1 of the best
checkpoints; a gap below TIE_F1 (0.5 point, about 4-5 val images) is a tie and
the lower val loss wins. Test metrics are listed for the record only.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
TIE_F1 = 0.005


def load_runs(output_root):
    runs = []
    for path in sorted(Path(output_root).glob("bblr_*/seed_*/*/metrics.json")):
        metrics = json.loads(path.read_text())
        runs.append(
            {
                "backbone": metrics["backbone"],
                "seed": metrics["seed"],
                "backbone_lr": metrics["backbone_lr"],
                "head_lr": metrics["head_lr"],
                "head_selector": metrics["head_selector"],
                "best_epoch": metrics["best_epoch"],
                "epochs_trained": metrics["epochs_trained"],
                "val_f1_macro": metrics["val"]["f1_macro"],
                "val_accuracy": metrics["val"]["accuracy"],
                "val_loss": metrics["val"]["loss"],
                "test_f1_macro": metrics["test"]["f1_macro"],
                "test_accuracy": metrics["test"]["accuracy"],
                "checkpoint": metrics["checkpoint"],
            }
        )
    return runs


def choose(runs, tie=TIE_F1):
    """Return one decision per (backbone, seed) that has exactly two LRs."""
    groups = {}
    for run in runs:
        groups.setdefault((run["backbone"], run["seed"]), []).append(run)
    decisions = []
    for (backbone, seed), group in sorted(groups.items()):
        if len(group) != 2 or len({r["backbone_lr"] for r in group}) != 2:
            decisions.append(
                {"backbone": backbone, "seed": seed, "chosen_backbone_lr": None,
                 "reason": f"incomplete: {len(group)} run(s)"}
            )
            continue
        a, b = sorted(group, key=lambda r: r["val_f1_macro"], reverse=True)
        gap = a["val_f1_macro"] - b["val_f1_macro"]
        if gap >= tie:
            winner, reason = a, f"val macro-F1 higher by {gap * 100:.2f} pt"
        else:
            winner = min(group, key=lambda r: r["val_loss"])
            reason = (
                f"tie on val macro-F1 ({gap * 100:.2f} pt < {tie * 100:.1f}); "
                f"lower val loss ({winner['val_loss']:.4f})"
            )
        decisions.append(
            {"backbone": backbone, "seed": seed,
             "chosen_backbone_lr": winner["backbone_lr"], "reason": reason}
        )
    return decisions


def markdown(runs, decisions):
    lines = [
        "| Model | Backbone LR | Val F1 | Val acc | Val loss | Best epoch | Epochs | Test acc | Test F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(runs, key=lambda r: (r["backbone"], r["backbone_lr"])):
        lines.append(
            f"| {r['backbone']} | {r['backbone_lr']:g} | {r['val_f1_macro'] * 100:.2f} | "
            f"{r['val_accuracy'] * 100:.2f} | {r['val_loss']:.4f} | {r['best_epoch']} | "
            f"{r['epochs_trained']} | {r['test_accuracy'] * 100:.2f} | {r['test_f1_macro'] * 100:.2f} |"
        )
    lines += ["", "| Model | Chosen backbone LR | Reason |", "|---|---:|---|"]
    for d in decisions:
        chosen = "–" if d["chosen_backbone_lr"] is None else f"{d['chosen_backbone_lr']:g}"
        lines.append(f"| {d['backbone']} | {chosen} | {d['reason']} |")
    chosen = {d["chosen_backbone_lr"] for d in decisions}
    if decisions and None not in chosen and len(chosen) == 1:
        lines += ["", f"All models choose backbone LR {chosen.pop():g}: use it for steps 2-4."]
    return "\n".join(lines) + "\n"


def write_report(output_root, tie=TIE_F1):
    runs = load_runs(output_root)
    if not runs:
        raise ValueError(f"No completed runs (metrics.json) under {output_root}")
    decisions = choose(runs, tie)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = Path(output_root) / "reports" / f"lr-select-{stamp}"
    directory.mkdir(parents=True, exist_ok=False)
    for name, rows in (("runs.csv", runs), ("decisions.csv", decisions)):
        with (directory / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    text = markdown(runs, decisions)
    (directory / "summary.md").write_text(text, encoding="utf-8")
    return directory, text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output_root", type=Path, default=ROOT / "data/experiments/Task-B-lr-select"
    )
    args = parser.parse_args(argv)
    directory, text = write_report(args.output_root)
    print(text)
    print(f"Report: {directory}")


if __name__ == "__main__":
    main()
