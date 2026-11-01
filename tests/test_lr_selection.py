"""Step 1 learning-rate selection: a tiny real run and the selection rule."""

import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from PIL import Image

TASK_DIR = (
    Path(__file__).resolve().parents[1]
    / "notebooks/phase 2/supervisor-feedback/task_07_lr_selection"
)
sys.path.insert(0, str(TASK_DIR))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, TASK_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


select = _load("train_lr_select")
summary = _load("summarize_lr_select")


class TinyClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(4), torch.nn.Flatten(), torch.nn.Linear(48, 8)
        )
        self.fc = torch.nn.Linear(8, num_classes)

    def forward(self, images):
        return self.fc(self.body(images))

    def get_classifier(self):
        return self.fc


@pytest.fixture
def run(tmp_path, monkeypatch):
    data = tmp_path / "data"
    rng = np.random.default_rng(0)
    for split, count in (("train", 3), ("val", 2), ("test", 2)):
        for species in range(17):
            folder = data / split / f"Species {species:02d}"
            folder.mkdir(parents=True)
            for index in range(count):
                pixels = rng.integers(0, 256, (12, 12, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(folder / f"{index}.png")
    monkeypatch.setattr(
        select, "create_model", lambda name, num_classes, pretrained: TinyClassifier(num_classes)
    )
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    output = tmp_path / "runs"
    argv = [
        "--backbone", "convnext_tiny", "--data_root", str(data), "--output_root", str(output),
        "--epochs", "2", "--warmup_epochs", "0", "--batch_size", "8", "--num_workers", "0",
    ]
    return output, argv


@pytest.mark.parametrize("lr,tag", [(1e-5, "1e-5"), (1e-4, "1e-4"), (3e-4, "3e-4")])
def test_lr_tag_names_output_folders(lr, tag):
    assert select.lr_tag(lr) == tag


def test_run_scores_val_and_test_from_best_checkpoint_and_refuses_resume(run):
    output, argv = run
    metrics = select.main(argv + ["--backbone_lr", "1e-5"])
    folder = output / "bblr_1e-5/seed_42/convnext_tiny"
    config = json.loads((folder / "run_config.json").read_text())
    assert config["head_selector"] == "classifier"
    assert config["head_params"] == 8 * 17 + 17
    assert config["backbone_lr"] == 1e-5 and config["head_lr"] == 1e-4
    assert config["split_sizes"] == {"train": 51, "val": 34, "test": 34}
    best = torch.load(folder / "model_best.pth", map_location="cpu", weights_only=False)
    assert metrics["best_epoch"] == best["best_epoch"] + 1
    assert metrics["val"]["n"] == 34 and metrics["test"]["n"] == 34
    rows = list(csv.DictReader((folder / "predictions_val.csv").open()))
    assert len(rows) == 34
    accuracy = np.mean([r["true_label"] == r["predicted_label"] for r in rows])
    assert metrics["val"]["accuracy"] == pytest.approx(accuracy)
    assert len(json.loads((folder / "history.json").read_text())) == metrics["epochs_trained"]
    with pytest.raises(FileExistsError, match="not resumed"):
        select.main(argv + ["--backbone_lr", "1e-5"])


def test_dryrun_writes_nothing(run):
    output, argv = run
    config = select.main(argv + ["--backbone_lr", "1e-4", "--dryrun"])
    assert config["head_params"] == 8 * 17 + 17
    assert not output.exists()


def _run(backbone, lr, f1, loss):
    return {
        "backbone": backbone, "seed": 42, "backbone_lr": lr, "val_f1_macro": f1,
        "val_loss": loss,
    }


def test_selection_uses_val_f1_then_val_loss_on_ties():
    decisions = {
        d["backbone"]: d
        for d in summary.choose(
            [
                _run("resnet50", 1e-5, 0.90, 0.70),
                _run("resnet50", 1e-4, 0.96, 0.80),
                _run("convnext_tiny", 1e-5, 0.997, 0.55),
                _run("convnext_tiny", 1e-4, 0.999, 0.60),
                _run("swin_tiny", 1e-4, 0.99, 0.6),
            ]
        )
    }
    assert decisions["resnet50"]["chosen_backbone_lr"] == 1e-4
    assert "higher" in decisions["resnet50"]["reason"]
    # 0.2 pt apart is a tie, so the lower val loss (1e-5) wins.
    assert decisions["convnext_tiny"]["chosen_backbone_lr"] == 1e-5
    assert "tie" in decisions["convnext_tiny"]["reason"]
    assert decisions["swin_tiny"]["chosen_backbone_lr"] is None


def test_report_from_completed_runs(run):
    output, argv = run
    for lr in ("1e-5", "1e-4"):
        select.main(argv + ["--backbone_lr", lr])
    directory, text = summary.write_report(output)
    runs = list(csv.DictReader((directory / "runs.csv").open()))
    assert {float(r["backbone_lr"]) for r in runs} == {1e-5, 1e-4}
    decisions = list(csv.DictReader((directory / "decisions.csv").open()))
    assert len(decisions) == 1 and float(decisions[0]["chosen_backbone_lr"]) in (1e-5, 1e-4)
    assert "| convnext_tiny |" in text


def test_launcher_runs_every_model_lr_pair_and_reports_failures(tmp_path):
    runner = tmp_path / "python-stub"
    runner.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$*" >> "$CALLS"\n'
        '[[ "$*" == *"--backbone resnet50 --backbone_lr 1e-4"* ]] && exit 7\nexit 0\n'
    )
    runner.chmod(0o755)
    env = dict(
        os.environ,
        PYTHON=str(runner),
        LOG_ROOT=str(tmp_path / "logs"),
        CALLS=str(tmp_path / "calls"),
        MODELS="convnext_tiny resnet50",
        EPOCHS="3",
        MAX_PARALLEL="1",
    )
    for name in ("LRS", "SEED", "NUM_WORKERS", "OUTPUT_ROOT", "DATA_ROOT"):
        env.pop(name, None)
    result = subprocess.run(
        ["bash", str(TASK_DIR / "run_lr_select.sh")],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "resnet50_bblr1e-4 failed (exit 7)" in result.stderr
    calls = (tmp_path / "calls").read_text().splitlines()
    assert len(calls) == 4
    assert all(str(TASK_DIR / "train_lr_select.py") in c and "--epochs 3" in c for c in calls)
    assert all("--seed" not in c and "--output_root" not in c for c in calls)
    assert {c.split("--backbone_lr ")[1].split()[0] for c in calls} == {"1e-5", "1e-4"}
    exits = sorted(p.read_text().strip() for p in (tmp_path / "logs").glob("*/*.exit"))
    assert exits == ["0", "0", "0", "7"]
