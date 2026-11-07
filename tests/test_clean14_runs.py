"""Supervisor requests 6-7 (06-11-2026): 14-species CV and Final-B."""

import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from tests.test_origin_cv import TASK_DIR, cv, origins
from tests.test_cv_training_runs import training  # noqa: F401  (fixture)

CLEAN14_DIR = TASK_DIR.parent / "task_15_clean14_runs"
ROOT = TASK_DIR.parents[3]


def _three_species(tmp_path):
    data, rows = tmp_path / "data", []
    for species in ["A", "B", "C"]:
        for origin in range(10):
            split = ["train", "val", "test"][origin % 3]
            name = f"sample.{origin}.(0).png"
            path = data / split / species / name
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), (origin * 20, 0, 0)).save(path)
            rows.append({"new_split": split, "old_split": split, "class_name": species,
                         "filename": name, "origin_key": f"{species}::sample.{origin}.",
                         "md5": f"{species}{origin}".ljust(32, "0")})
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return data, manifest


def test_drop_species_keeps_remaining_folds_and_renumbers_classes(tmp_path):
    data, manifest = _three_species(tmp_path)
    full = cv.OriginKFoldDataset(data, "all", manifest_path=manifest)
    splits = full.get_splits_by_origin(5, seed=42, inner_val_fraction=0.2)
    before = [
        {key: {o for o in s[key] if not o.startswith("A::")}
         for key in ("val_origins", "inner_val_origins", "inner_train_origins")}
        for s in splits
    ]
    dataset, dropped = origins.drop_species(full, splits, ["A"])
    assert dataset.classes == ["B", "C"] and dataset.class_to_idx == {"B": 0, "C": 1}
    assert len(dataset) == 20
    for split, expected in zip(dropped, before):
        for key, value in expected.items():
            assert split[key] == value
        assert {dataset.records[i]["origin_key"] for i in split["val_idx"]} == split["val_origins"]
        assert {dataset.records[i]["origin_key"] for i in split["inner_val_idx"]} == split["inner_val_origins"]
        assert split["n_val"] == len(split["val_idx"])
        assert sorted(np.concatenate([split["inner_train_idx"], split["inner_val_idx"]])) == sorted(split["train_idx"])
    assert all(label == dataset.class_to_idx[r["species"]]
               for (_, label, _), r in zip(dataset.samples, dataset.records))
    with pytest.raises(ValueError, match="Unknown species"):
        origins.drop_species(dataset, dropped, ["Z"])


def test_clean14_set_names_three_existing_species():
    assert set(origins.SPECIES_SETS["clean14"]) == {
        "Callitris columellaris", "Cunninghamia lanceolata", "Fokienia hodginsii"}
    data = ROOT / "data/task-B-group-split-224"
    if not data.exists():
        pytest.skip("224 cache not installed")
    assert set(origins.SPECIES_SETS["clean14"]) <= {p.name for p in (data / "train").iterdir()}


def test_cv_species_set_trains_on_subset_and_records_it(training, monkeypatch):  # noqa: F811
    output, argv, _ = training
    monkeypatch.setitem(cv.SPECIES_SETS, "noA", ("A",))
    cv.main(argv + ["--fold", "1", "--species_set", "noA"])
    metrics = json.loads((output / "convnext_tiny/fold_1/metrics.json").read_text())
    assert metrics["config"]["species_set"] == "noA"
    assert metrics["config"]["excluded_species"] == ["A"]
    assert metrics["class_to_idx"] == {"B": 0}
    rows = list(csv.DictReader((output / "convnext_tiny/fold_1/predictions.csv").open()))
    assert rows and {r["true_species"] for r in rows} == {"B"}


def test_cv_default_species_set_keeps_config_keys(training):  # noqa: F811
    output, argv, _ = training
    cv.main(argv + ["--fold", "1"])
    config = json.loads((output / "convnext_tiny/fold_1/metrics.json").read_text())["config"]
    assert "species_set" not in config and "excluded_species" not in config


def test_launcher_queues_all_runs_with_the_new_rule(tmp_path):
    runner = tmp_path / "runner"
    runner.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['CALLS'], 'a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    runner.chmod(0o755)
    env = dict(os.environ, PYTHON=str(runner), CALLS=str(tmp_path / "calls"),
               EXP_ROOT=str(tmp_path / "exp"), SKIP_DATA_CHECK="1", MAX_PARALLEL="3")
    result = subprocess.run(["bash", str(CLEAN14_DIR / "run_clean14.sh")], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr + result.stdout
    calls = [json.loads(line) for line in (tmp_path / "calls").read_text().splitlines()]
    calls = [c for c in calls if c[:1] != ["-m"]]
    cv_calls = [c for c in calls if c[0].endswith("train_kfold.py")]
    final_calls = [c for c in calls if c[0].endswith("train_final_b.py")]
    assert len(cv_calls) == 7 and len(final_calls) == 15

    def option(call, name):
        return call[call.index(name) + 1]

    for call in cv_calls:
        assert option(call, "--species_set") == "clean14"
        assert option(call, "--backbone_lr") == "1e-4" and option(call, "--head_lr") == "1e-4"
        assert option(call, "--selection_metric") == "val_loss"
        assert option(call, "--early_stopping_patience") == "15"
        assert {"--select_after_warmup", "--resume"} <= set(call)
    variants = {(option(c, "--backbone"), option(c, "--input_variant") if "--input_variant" in c else "rgb",
                 Path(option(c, "--output_root")).name) for c in cv_calls}
    assert ("convnext_tiny", "grayscale", "Task-B-clean14-cv-grayscale") in variants
    assert ("convnext_tiny", "px16", "Task-B-clean14-cv-px16") in variants
    assert {v[0] for v in variants if v[1] == "rgb"} == set(cv.BACKBONE_NAMES)
    assert {(option(c, "--backbone"), option(c, "--seed")) for c in final_calls} == {
        (b, s) for b in cv.BACKBONE_NAMES for s in ("42", "123", "456")}
    logs = next((tmp_path / "exp/clean14-logs").iterdir())
    assert len(list(logs.glob("*.exit"))) == 22


def _final_b():
    spec = importlib.util.spec_from_file_location("train_final_b", CLEAN14_DIR / "train_final_b.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_final_b_data_check_rejects_changed_missing_or_extra_images(tmp_path):
    final_b = _final_b()
    data = tmp_path / "task-B-group-split-224"
    for name in ("train/A/1.png", "val/A/2.png"):
        (data / name).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4)).save(data / name)
    checksums = tmp_path / "list.sha256"
    checksums.write_text("".join(
        f"{final_b.file_sha256(data / n)}  task-B-group-split-224/{n}\n"
        for n in ("train/A/1.png", "val/A/2.png")))
    assert final_b.verify_data(data, checksums)
    Image.new("RGB", (4, 4), (1, 2, 3)).save(data / "val/A/2.png")
    with pytest.raises(ValueError, match="differ"):
        final_b.verify_data(data, checksums)
    (data / "val/A/2.png").unlink()
    with pytest.raises(ValueError, match="missing"):
        final_b.verify_data(data, checksums)


def test_final_b_defaults_are_the_06_11_rule():
    args = _final_b().build_parser().parse_args(["--backbone", "resnet50", "--seed", "42"])
    assert (args.backbone_lr, args.head_lr, args.warmup_epochs, args.epochs) == (1e-4, 1e-4, 5, 50)
    assert args.early_stopping_patience == 15 and args.species_set == "clean14"
    assert args.data_root.name == "task-B-group-split-224"
