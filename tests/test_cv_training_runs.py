"""Small real CPU runs exercise artifacts, fold restart and reporting."""

import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import statistics

import pytest
import torch
from torchvision import transforms

from tests.test_origin_cv import TASK_DIR, _dataset, cv


@pytest.mark.parametrize("backbone", list(cv.BACKBONE_NAMES))
def test_default_training_budget_matches_phase1_per_backbone(
    training, monkeypatch, backbone
):
    import ast

    phase1 = cv.ROOT / "notebooks/phase 1/task-B-17-new/final_b_17class.py"
    batch_config = next(
        ast.literal_eval(node.value)
        for node in ast.parse(phase1.read_text()).body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "BACKBONE_CONFIGS"
    )
    _, argv, _ = training
    options = dict(zip(argv[::2], argv[1::2]))
    options.pop("--epochs")
    options.pop("--batch_size")
    options["--backbone"] = backbone
    captured = []
    save_checkpoint = cv.save_checkpoint

    def capture(path, checkpoint):
        captured.append(checkpoint["config"])
        save_checkpoint(path, checkpoint)

    monkeypatch.setattr(cv, "save_checkpoint", capture)
    cv.main([value for pair in options.items() for value in pair] + ["--dryrun"])
    assert captured[0]["epochs"] == 50
    assert captured[0]["batch_size"] == batch_config[backbone]["batch_size"]


def test_notebook_uses_main_budget_and_defers_batch_choice_to_engine(tmp_path):
    notebook = json.loads((TASK_DIR / "fivefold_cv_training.ipynb").read_text())
    namespace = {"ROOT": tmp_path}
    config_source = next(
        "".join(c["source"])
        for c in notebook["cells"]
        if "CONFIG = {" in "".join(c["source"])
    )
    exec(config_source, namespace)
    assert namespace["CONFIG"]["epochs"] == 50
    assert namespace["CONFIG"]["batch_size"] is None
    assert set(namespace["CONFIG"]["models"]) == set(cv.BACKBONE_NAMES)
    source = next(
        "".join(c["source"])
        for c in notebook["cells"]
        if "summaries.append(train_cv(argv))" in "".join(c["source"])
    )
    calls = []
    fake = type(sys)("train_kfold")
    fake.main = lambda args: calls.append(args)
    from unittest.mock import patch

    with patch.dict(sys.modules, {"train_kfold": fake}):
        exec(source, namespace)
    assert all("--batch_size" not in call for call in calls)
    assert all(call[call.index("--epochs") + 1] == "50" for call in calls)


@pytest.mark.parametrize("batch_override", [None, "16"])
def test_launcher_defers_defaults_and_forwards_explicit_overrides(
    tmp_path, batch_override
):
    runner = tmp_path / "runner"
    runner.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['CALLS'], 'a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    runner.chmod(0o755)
    env = dict(os.environ)
    for name in ("BATCH_SIZE", "EPOCHS", "BACKBONE_LR"):
        env.pop(name, None)
    env.update(
        PYTHON=str(runner),
        LOG_ROOT=str(tmp_path / "logs"),
        CALLS=str(tmp_path / "calls"),
        MODELS="resnet50",
        RESUME="1",
        OUTPUT_ROOT=str(tmp_path / "runs"),
    )
    if batch_override:
        env.update(BATCH_SIZE=batch_override, EPOCHS="7", BACKBONE_LR="0.0001")
    result = subprocess.run(
        ["bash", str(TASK_DIR / "run_all_models.sh")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    argv = json.loads((tmp_path / "calls").read_text())
    assert "--resume" in argv
    assert argv[argv.index("--output_root") + 1] == str(tmp_path / "runs")
    if batch_override:
        assert argv[argv.index("--batch_size") + 1] == "16"
        assert argv[argv.index("--epochs") + 1] == "7"
        assert argv[argv.index("--backbone_lr") + 1] == "0.0001"
    else:
        assert {"--batch_size", "--epochs", "--backbone_lr"}.isdisjoint(argv)


def test_monitor_displays_completed_epochs_against_training_budget(training):
    output, argv, _ = training
    cv.main(argv + ["--fold", "1"])
    result = subprocess.run(
        [
            sys.executable,
            str(TASK_DIR / "check_progress.py"),
            "--output",
            str(output),
            "--model",
            "convnext_tiny",
            "--verbose",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "epoch=2/2" in result.stdout


def test_every_fold_uses_the_same_seed_recorded_in_its_run_config(
    training, monkeypatch
):
    output, argv, _ = training
    seeds = []
    set_seed = cv.set_seed

    def record(seed):
        seeds.append(seed)
        set_seed(seed)

    monkeypatch.setattr(cv, "set_seed", record)
    cv.main(argv)
    recorded = [
        json.loads((output / f"convnext_tiny/fold_{fold}/metrics.json").read_text())[
            "config"
        ]["seed"]
        for fold in range(1, 6)
    ]
    assert seeds == recorded == [42] * 5


@pytest.mark.parametrize("use_90rotation", [False, True])
@pytest.mark.parametrize("seed", [42, 123, 456])
def test_training_augmentation_matches_phase1_for_identical_image_and_seed(
    use_90rotation, seed
):
    import numpy as np
    from PIL import Image
    from src.data.transforms import get_train_transforms

    image = Image.fromarray(
        np.random.default_rng(7).integers(0, 256, (96, 160, 3), dtype=np.uint8)
    )
    train, _ = cv.get_transforms(use_90rotation=use_90rotation)
    cv.set_seed(seed)
    actual = train(image)
    cv.set_seed(seed)
    expected = get_train_transforms(use_90rotation=use_90rotation)(image)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_held_out_preprocessing_matches_phase1_without_cropping():
    import numpy as np
    from PIL import Image
    from src.data.transforms import get_val_transforms

    image = Image.fromarray(
        np.random.default_rng(9).integers(0, 256, (96, 160, 3), dtype=np.uint8)
    )
    _, held_out = cv.get_transforms()
    torch.testing.assert_close(
        held_out(image), get_val_transforms()(image), rtol=0, atol=0
    )


def test_notebook_defaults_to_phase1_main_augmentation(tmp_path):
    notebook = json.loads((TASK_DIR / "fivefold_cv_training.ipynb").read_text())
    source = next(
        "".join(c["source"])
        for c in notebook["cells"]
        if "CONFIG = {" in "".join(c["source"])
    )
    namespace = {"ROOT": tmp_path}
    exec(source, namespace)
    assert namespace["CONFIG"]["use_90rotation"] is False


def test_resume_identity_includes_shared_phase1_transform_code(tmp_path, monkeypatch):
    import run_artifacts as artifacts

    data, manifest, _ = _dataset(tmp_path)
    dataset = cv.OriginKFoldDataset(data, "all", manifest_path=manifest)
    args = cv.build_parser().parse_args(
        ["--backbone", "convnext_tiny", "--manifest_path", str(manifest)]
    )
    splits = dataset.get_splits_by_origin(5, seed=42)
    original = artifacts.make_config(args, dataset, splits)
    hash_file = artifacts.file_sha256

    def changed_transform(path):
        if Path(path) == cv.ROOT / "src/data/transforms.py":
            return "modified-transform-source"
        return hash_file(path)

    monkeypatch.setattr(artifacts, "file_sha256", changed_transform)
    changed = artifacts.make_config(args, dataset, splits)
    assert original["code_sha256"] != changed["code_sha256"]


@pytest.mark.parametrize("use_90rotation", [False, True])
def test_registry_records_phase1_augmentation(training, use_90rotation):
    from cv_reports import export_reports

    output, argv, _ = training
    cv.main(argv + (["--use_90rotation"] if use_90rotation else []))
    registry = export_reports(output)
    rows = list(csv.DictReader(registry.open()))
    rotation = "random_90" if use_90rotation else "random_rotation30"
    expected = f"resize224|hflip0.5|vflip0.5|{rotation}|color_jitter0.2"
    assert {row["augmentation"] for row in rows} == {expected}
    assert {
        json.loads((output / f"convnext_tiny/fold_{fold}/metrics.json").read_text())[
            "config"
        ]["augmentation"]
        for fold in range(1, 6)
    } == {expected}


@pytest.fixture
def training(tmp_path, monkeypatch):
    data, manifest, _ = _dataset(tmp_path)
    output = tmp_path / "runs"
    calls = []

    def create_model(*args, **kwargs):
        calls.append("model")
        return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 2))

    monkeypatch.setattr(cv.timm, "create_model", create_model)
    transform = transforms.Compose([transforms.Resize((8, 8)), transforms.ToTensor()])
    monkeypatch.setattr(cv, "get_transforms", lambda **kwargs: (transform, transform))
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    argv = [
        "--backbone",
        "convnext_tiny",
        "--data_root",
        str(data),
        "--manifest_path",
        str(manifest),
        "--output_root",
        str(output),
        "--epochs",
        "2",
        "--batch_size",
        "32",
        "--num_workers",
        "0",
    ]
    return output, argv, calls


def test_fixed_epochs_evaluate_once_and_write_safe_traceable_artifacts(
    training, monkeypatch
):
    output, argv, calls = training
    evaluations = []
    original = cv.evaluate_full

    def evaluate(*args, **kwargs):
        evaluations.append(len(calls))
        return original(*args, **kwargs)

    monkeypatch.setattr(cv, "evaluate_full", evaluate)
    cv.main(argv + ["--fold", "1"])
    folder = output / "convnext_tiny/fold_1"
    checkpoint = torch.load(folder / "final.pth", map_location="cpu")
    assert checkpoint["epoch"] == 1
    assert "optimizer_state_dict" not in checkpoint
    assert "scheduler_state_dict" not in checkpoint
    assert evaluations == [1]
    assert not (folder / "best.pth").exists()
    metrics = json.loads((folder / "metrics.json").read_text())
    assert metrics["run_id"] == checkpoint["run_id"]
    assert metrics["protocol"] == "fixed-epochs-origin-cv-v1"
    assert metrics["class_to_idx"] == {"A": 0, "B": 1}
    assert isinstance(metrics["config"]["manifest_path"], str)
    rows = list(csv.DictReader((folder / "predictions.csv").open()))
    assert len(rows) == metrics["n_test"] == 8
    assert {r["origin_key"] for r in rows}.isdisjoint(metrics["train_origins"])
    assert {r["run_id"] for r in rows} == {metrics["run_id"]}


def test_resume_keeps_completed_folds_and_restarts_only_unfinished_fold(training):
    output, argv, calls = training
    cv.main(argv + ["--fold", "1"])
    done = output / "convnext_tiny/fold_1/metrics.json"
    original = done.read_bytes()
    unfinished = output / "convnext_tiny/fold_2"
    unfinished.mkdir()
    (unfinished / "last.pth").write_bytes(b"interrupted checkpoint")
    cv.main(argv + ["--resume"])
    assert done.read_bytes() == original
    assert len(calls) == 5
    assert all(
        (output / f"convnext_tiny/fold_{i}/metrics.json").exists() for i in range(1, 6)
    )
    assert torch.load(unfinished / "final.pth", map_location="cpu")["epoch"] == 1
    assert not (unfinished / "last.pth").exists()


@pytest.mark.parametrize("changed", [["--seed", "123"], ["--epochs", "3"]])
def test_resume_rejects_changed_config_without_overwriting_results(training, changed):
    output, argv, calls = training
    cv.main(argv + ["--fold", "1"])
    metrics = output / "convnext_tiny/fold_1/metrics.json"
    original = metrics.read_bytes()
    with pytest.raises(ValueError, match="config|fingerprint|different"):
        cv.main(argv + ["--resume"] + changed)
    assert len(calls) == 1
    assert metrics.read_bytes() == original


def test_resume_rejects_legacy_metrics_and_missing_artifacts(training):
    output, argv, _ = training
    folder = output / "convnext_tiny/fold_1"
    folder.mkdir(parents=True)
    metrics = folder / "metrics.json"
    metrics.write_text('{"best_val_acc": 0.99}')
    with pytest.raises(ValueError, match="legacy|fingerprint|different"):
        cv.main(argv + ["--resume"])
    metrics.unlink()
    cv.main(argv + ["--fold", "1", "--resume"])
    (folder / "predictions.csv").unlink()
    with pytest.raises(ValueError, match="artifact|missing"):
        cv.main(argv + ["--resume"])


def test_fresh_run_does_not_silently_overwrite_completed_fold(training):
    _, argv, calls = training
    cv.main(argv + ["--fold", "1"])
    with pytest.raises((FileExistsError, ValueError), match="resume|exists"):
        cv.main(argv + ["--fold", "1"])
    assert len(calls) == 1


def test_fold_restart_reproduces_initial_model_and_final_weights(training):
    output, argv, _ = training
    cv.main(argv + ["--fold", "1"])
    folder = output / "convnext_tiny/fold_1"
    state = torch.load(folder / "final.pth", map_location="cpu")["model_state_dict"]
    (folder / "metrics.json").unlink()
    cv.main(argv + ["--fold", "1", "--resume"])
    repeated = torch.load(folder / "final.pth", map_location="cpu")["model_state_dict"]
    assert all(torch.equal(state[k], repeated[k]) for k in state)


def test_real_interruption_keeps_completed_fold_and_retrains_failed_fold(
    training, monkeypatch
):
    output, argv, calls = training
    cv.main(argv + ["--fold", "1"])
    original_train = cv.train_epoch
    epochs = []

    def interrupted(*args, **kwargs):
        epochs.append(1)
        if len(epochs) == 2:
            raise RuntimeError("simulated interruption")
        return original_train(*args, **kwargs)

    monkeypatch.setattr(cv, "train_epoch", interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        cv.main(argv + ["--fold", "2", "--resume"])
    folder = output / "convnext_tiny/fold_2"
    partial = json.loads((folder / "run.json").read_text())
    assert partial["epoch"] == 1
    assert partial["status"] == "interrupted"
    assert not (folder / "last.pth").exists()
    assert not (folder / "metrics.json").exists()
    restarted_epochs = []

    def restarted(*args, **kwargs):
        restarted_epochs.append(1)
        return original_train(*args, **kwargs)

    monkeypatch.setattr(cv, "train_epoch", restarted)
    cv.main(argv + ["--fold", "2", "--resume"])
    assert len(restarted_epochs) == 2
    assert len(calls) == 3
    assert (
        torch.load(folder / "final.pth", map_location="cpu")["run_id"]
        != partial["run_id"]
    )


def test_resume_rejects_swapped_fold_artifacts(training):
    import shutil

    output, argv, _ = training
    cv.main(argv + ["--fold", "1"])
    shutil.copytree(output / "convnext_tiny/fold_1", output / "convnext_tiny/fold_2")
    with pytest.raises(ValueError, match="fold|identity"):
        cv.main(argv + ["--resume"])


def test_model_output_is_locked_against_concurrent_training(training):
    output, argv, calls = training
    from run_artifacts import model_lock

    with model_lock(output / "convnext_tiny"):
        with pytest.raises(RuntimeError, match="already owns"):
            cv.main(argv)
    assert not calls


def test_reports_reject_partial_cv(training):
    output, argv, _ = training
    cv.main(argv + ["--fold", "1"])
    from cv_reports import export_reports

    with pytest.raises(ValueError, match="Incomplete"):
        export_reports(output)
    assert not (output / "reports").exists()


def test_single_fold_uses_shared_restart_and_evaluation_protocol(training):
    _, argv, _ = training
    result = cv.main(argv + ["--fold=1"])
    assert result["n_completed"] == 1
    with pytest.raises(ValueError, match="Fold must be"):
        cv.main(argv + ["--fold", "6"])


def test_registry_is_immutable_and_reports_link_to_actual_runs(training):
    output, argv, _ = training
    run_summary = cv.main(argv)
    from cv_reports import export_reports

    first = export_reports(output)
    before = first.read_bytes()
    second = export_reports(output)
    assert first != second
    assert first.read_bytes() == before
    rows = list(csv.DictReader(first.open()))
    assert len(rows) == 5
    from sklearn.metrics import classification_report

    for row in rows:
        folder = output / f"convnext_tiny/fold_{row['fold']}"
        predictions = list(csv.DictReader((folder / "predictions.csv").open()))
        expected = classification_report(
            [int(p["true_label"]) for p in predictions],
            [int(p["predicted_label"]) for p in predictions],
            output_dict=True,
            zero_division=0,
        )
        assert float(row["test_precision_macro"]) == pytest.approx(
            expected["macro avg"]["precision"]
        )
        assert float(row["test_recall_macro"]) == pytest.approx(
            expected["macro avg"]["recall"]
        )
        assert float(row["test_f1_macro"]) == pytest.approx(
            expected["macro avg"]["f1-score"]
        )
    assert len({r["run_id"] for r in rows}) == 5
    required = {
        "run_date",
        "notebook",
        "git_commit",
        "split_version",
        "num_classes",
        "n_train",
        "n_val",
        "n_test",
        "seed",
        "lr",
        "epochs",
        "batch",
        "image_size",
        "label_smoothing",
        "augmentation",
        "checkpoint_path",
        "prediction_path",
    }
    assert required <= rows[0].keys()
    assert all(not Path(r["checkpoint_path"]).is_absolute() for r in rows)
    report = list(csv.DictReader((first.parent / "per_class.csv").open()))
    assert len(report) == 10
    assert {r["run_id"] for r in report} == {r["run_id"] for r in rows}
    summary = list(csv.DictReader((first.parent / "summary.csv").open()))
    assert len(json.loads(summary[0]["source_run_ids"])) == 5
    assert json.loads(summary[0]["source_run_ids"]) == run_summary["source_run_ids"]
    assert float(summary[0]["test_accuracy_std"]) == pytest.approx(
        statistics.stdev(float(row["test_accuracy"]) for row in rows)
    )
    oof = list(csv.DictReader((first.parent / "oof_predictions.csv").open()))
    assert len(oof) == 40
    assert len({r["path"] for r in oof}) == 40


def test_missing_output_monitor_does_not_crash(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(TASK_DIR / "check_progress.py"),
            "--output",
            str(tmp_path / "missing"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "0/25" in result.stdout


def test_monitor_distinguishes_interrupted_and_zero_accuracy(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "cv_monitor", TASK_DIR / "check_progress.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = tmp_path / "model"
    (model / "fold_1").mkdir(parents=True)
    (model / "fold_1/last.pth").write_bytes(b"partial")
    (model / "fold_2").mkdir()
    (model / "fold_2/metrics.json").write_text(
        json.dumps({"status": "completed", "accuracy": 0.0})
    )
    progress = module.get_model_progress(model)
    assert progress["folds"][0]["status"] == "interrupted"
    assert progress["accuracies"] == [0.0]


def test_monitor_marks_missing_completed_artifact_as_error(training):
    output, argv, _ = training
    cv.main(argv + ["--fold", "1"])
    (output / "convnext_tiny/fold_1/final.pth").unlink()
    result = subprocess.run(
        [
            sys.executable,
            str(TASK_DIR / "check_progress.py"),
            "--output",
            str(output),
            "--model",
            "convnext_tiny",
            "--verbose",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "completed=0" in result.stdout and "error=1" in result.stdout


def test_monitor_reports_corrupt_config_instead_of_crashing(tmp_path):
    model = tmp_path / "runs/convnext_tiny"
    model.mkdir(parents=True)
    (model / "run_config.json").write_text("{broken")
    result = subprocess.run(
        [
            sys.executable,
            str(TASK_DIR / "check_progress.py"),
            "--output",
            str(model.parent),
            "--model",
            "convnext_tiny",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "error=5" in result.stdout


def test_launcher_tracks_failure_limits_concurrency_and_preserves_logs(tmp_path):
    runner = tmp_path / "python-stub"
    runner.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$CALLS"\nexit 23\n')
    runner.chmod(0o755)
    logs = tmp_path / "logs"
    logs.mkdir()
    old = logs / "convnext_tiny.log"
    old.write_text("prior log")
    env = dict(
        os.environ,
        PYTHON=str(runner),
        LOG_ROOT=str(logs),
        CALLS=str(tmp_path / "calls"),
        MODELS="convnext_tiny mobilenetv3_large",
        MAX_PARALLEL="1",
    )
    result = subprocess.run(
        ["bash", str(TASK_DIR / "run_all_models.sh")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert old.read_text() == "prior log"
    calls = (tmp_path / "calls").read_text().splitlines()
    assert len(calls) == 2
    assert all(str(TASK_DIR / "train_kfold.py") in c for c in calls)
    assert len(list(logs.glob("*/convnext_tiny.log"))) == 1


@pytest.mark.parametrize("limit", [None, "2"])
def test_launcher_honors_maximum_concurrent_children(tmp_path, limit):
    runner = tmp_path / "runner"
    runner.write_text(f"""#!{sys.executable}
import fcntl, json, os, time
from pathlib import Path
path = Path(os.environ['COUNTER'])
def change(delta):
    with path.open('r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        values = json.load(f)
        values['active'] += delta
        values['peak'] = max(values['active'], values['peak'])
        f.seek(0); json.dump(values, f); f.truncate()
change(1)
time.sleep(0.3)
change(-1)
""")
    runner.chmod(0o755)
    counter = tmp_path / "counter"
    counter.write_text('{"active": 0, "peak": 0}')
    env = dict(
        os.environ,
        PYTHON=str(runner),
        LOG_ROOT=str(tmp_path / "logs"),
        COUNTER=str(counter),
        MODELS="convnext_tiny mobilenetv3_large resnet50 efficientnet_b3 swin_tiny",
    )
    env.pop("MAX_PARALLEL", None)
    if limit:
        env["MAX_PARALLEL"] = limit
    result = subprocess.run(
        ["bash", str(TASK_DIR / "run_all_models.sh")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(counter.read_text()) == {
        "active": 0,
        "peak": int(limit) if limit else 5,
    }


def test_notebook_calls_shared_engine_and_separate_report_export():
    notebook = json.loads((TASK_DIR / "fivefold_cv_training.ipynb").read_text())
    namespace = {
        "CONFIG": {
            "models": ["convnext_tiny"],
            "n_folds": 5,
            "epochs": 2,
            "batch_size": 2,
            "backbone_lr": 1e-5,
            "head_lr": 1e-4,
            "weight_decay": 1e-4,
            "label_smoothing": 0.1,
            "seed": 42,
            "num_workers": 0,
            "use_90rotation": True,
            "resume": True,
        },
        "DATA_DIR": Path("data"),
        "OUTPUT_DIR": Path("runs"),
    }
    source = next(
        "".join(c["source"])
        for c in notebook["cells"]
        if "summaries.append(train_cv(argv))" in "".join(c["source"])
    )
    calls = []
    fake = type(sys)("train_kfold")
    fake.main = lambda args: calls.append(args)
    from unittest.mock import patch

    with patch.dict(sys.modules, {"train_kfold": fake}):
        exec(source, namespace)
    assert "--resume" in calls[0] and "--use_90rotation" in calls[0]
    report = json.loads((TASK_DIR / "evaluate_and_register_cv.ipynb").read_text())
    assert any(
        "export_reports(OUTPUT_ROOT)" in "".join(c["source"]) for c in report["cells"]
    )
