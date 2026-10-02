"""Task 1 must reuse the origin identities of the final Phase 1 dataset."""

import csv
import ast
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from torchvision import transforms

TASK_DIR = (
    Path(__file__).resolve().parents[1]
    / "notebooks/phase 2/supervisor-feedback/task_01_fivefold_cv_training"
)
sys.path.insert(0, str(TASK_DIR))
import origin_cv as origins

spec = importlib.util.spec_from_file_location(
    "cv_training", TASK_DIR / "train_kfold.py"
)
cv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cv)


@pytest.mark.parametrize(
    "filename,expected",
    [
        (
            "3351. Araucaria araucana.1.(12).png",
            "Araucaria araucana::3351. Araucaria araucana.1.",
        ),
        (
            "3351. Araucaria araucana.1 (12).png",
            "Araucaria araucana::3351. Araucaria araucana.1",
        ),
        ("10.png", "Araucaria araucana::10"),
    ],
)
def test_origin_keys_follow_phase1_notebook(filename, expected):
    assert origins.infer_origin_key(Path("Araucaria araucana") / filename) == expected


def _dataset(tmp_path):
    data = tmp_path / "data"
    rows = []
    for species in ["A", "B"]:
        for origin in range(10):
            source_split = ["train", "val", "test"][origin % 3]
            for crop in range(2):
                name = f"sample.{origin}.({crop}).png"
                path = data / source_split / species / name
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (32, 32), (origin * 10, crop * 20, 80)).save(path)
                rows.append(
                    {
                        "new_split": source_split,
                        "old_split": source_split,
                        "class_name": species,
                        "filename": name,
                        "origin_key": f"{species}::sample.{origin}.",
                        "md5": f"{species}{origin}{crop}".ljust(32, "0"),
                    }
                )
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return data, manifest, rows


def test_folds_use_all_original_splits_and_keep_each_species_origin_together(tmp_path):
    data, manifest, _ = _dataset(tmp_path)
    dataset = cv.OriginKFoldDataset(data, "all", manifest_path=manifest)
    assert len(dataset) == 40
    splits = dataset.get_splits_by_origin(5, seed=42)
    seen = []
    for split in splits:
        assert split["train_origins"].isdisjoint(split["val_origins"])
        for species in ["A", "B"]:
            assert (
                sum(key.startswith(f"{species}::") for key in split["val_origins"]) == 2
            )
        seen.extend(split["val_idx"])
    assert sorted(seen) == list(range(40))
    repeated = dataset.get_splits_by_origin(5, seed=42)
    assert all(
        np.array_equal(a["val_idx"], b["val_idx"]) for a, b in zip(splits, repeated)
    )
    assert [s["val_origins"] for s in splits] != [
        s["val_origins"] for s in dataset.get_splits_by_origin(5, seed=123)
    ]


def test_missing_entire_species_is_rejected_against_phase1_manifest(tmp_path):
    import shutil

    data, manifest, _ = _dataset(tmp_path)
    for split in ("train", "val", "test"):
        shutil.rmtree(data / split / "B")
    with pytest.raises(ValueError, match="missing from Phase 1 manifest dataset"):
        cv.OriginKFoldDataset(data, "all", manifest_path=manifest)


def test_manifest_allows_only_the_species_excluded_from_final_phase1(tmp_path):
    phase1 = TASK_DIR.parents[3] / "notebooks/phase 1/task-B-17-new/final_b_17class.py"
    excluded = next(
        ast.literal_eval(node.value)
        for node in ast.parse(phase1.read_text()).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "EXCLUDED_CLASSES" for t in node.targets
        )
    )
    data, manifest, rows = _dataset(tmp_path)
    for index, species in enumerate(excluded):
        rows.append(
            {
                "new_split": "train",
                "old_split": "train",
                "class_name": species,
                "filename": "excluded.(0).png",
                "origin_key": f"{species}::excluded.",
                "md5": f"excluded{index}".ljust(32, "0"),
            }
        )
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    dataset = cv.OriginKFoldDataset(data, "all", manifest_path=manifest)
    assert len(dataset) == 40
    assert dataset.classes == ["A", "B"]


def test_collision_renaming_keeps_original_phase1_origin(tmp_path):
    data = tmp_path / "data"
    folder = data / "train" / "A"
    folder.mkdir(parents=True)
    for name in ["10.png", "10__from-val__bbbbbbbb.png"]:
        Image.new("RGB", (32, 32)).save(folder / name)
    manifest = tmp_path / "manifest.csv"
    rows = [
        {
            "new_split": "train",
            "old_split": old,
            "class_name": "A",
            "filename": "10.png",
            "origin_key": "A::10",
            "md5": digest * 32,
        }
        for old, digest in [("train", "a"), ("val", "b")]
    ]
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    dataset = cv.OriginKFoldDataset(data, "train", manifest_path=manifest)
    assert len(dataset) == 2
    assert {sample[2] for sample in dataset.samples} == {"A::10"}


def test_unmapped_image_is_rejected_instead_of_inventing_an_origin(tmp_path):
    data, manifest, _ = _dataset(tmp_path)
    Image.new("RGB", (32, 32)).save(data / "train/A/unknown.png")
    with pytest.raises(ValueError, match="manifest"):
        cv.OriginKFoldDataset(data, "all", manifest_path=manifest)


def test_fold_dataset_applies_image_transform_once(tmp_path):
    data, manifest, _ = _dataset(tmp_path)
    dataset = cv.OriginKFoldDataset(
        data, "all", transform=transforms.ToTensor(), manifest_path=manifest
    )
    subset = cv.SubsetWithTransform(dataset, [0], transforms.ToTensor())
    image, label = subset[0]
    assert image.shape == (3, 32, 32)
    assert label == 0


def test_original_dataset_origin_cannot_cross_source_splits(tmp_path):
    data, manifest, rows = _dataset(tmp_path)
    rows[0]["new_split"] = "val"
    original = data / "train/A/sample.0.(0).png"
    original.rename(data / "val/A/sample.0.(0).png")
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="source split"):
        cv.OriginKFoldDataset(data, "all", manifest_path=manifest)


def test_reuses_exact_phase1_key_function_on_manifest_filenames():
    root = TASK_DIR.parents[3]
    notebook = (
        root
        / "notebooks/phase 1/investigator/WoodID_Phase1_Leakage_Origin_Investigation.ipynb"
    )
    cells = json.loads(notebook.read_text())["cells"]
    source = next(
        "".join(c["source"])
        for c in cells
        if "def infer_origin_key(" in "".join(c["source"])
    )
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "infer_origin_key"
    )
    import re

    namespace = {"Path": Path, "re": re}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "phase1_origin", "exec"),
        namespace,
    )
    manifest = (
        root
        / "docs/reports/phase_1/phase_1_leakage_origin_investigation/task-B-clean_group_split_plan.csv"
    )
    for row in csv.DictReader(manifest.open()):
        path = Path(row["class_name"]) / row["filename"]
        assert (
            origins.infer_origin_key(path)
            == namespace["infer_origin_key"](path)
            == row["origin_key"]
        )


def test_real_dataset_shares_origin_folds_with_training_notebook():
    root = TASK_DIR.parents[3]
    data = root / "data/task-B-group-split"
    if not data.exists():
        pytest.skip("Local Phase 1 images are not installed")
    dataset = cv.OriginKFoldDataset(data, "all")
    assert len(dataset) == 4800
    assert len({s[2] for s in dataset.samples}) == 552
    splits = dataset.get_splits_by_origin(5, seed=42)
    for split in splits:
        assert {dataset.records[i]["species"] for i in split["val_idx"]} == set(
            dataset.classes
        )
        assert {dataset.records[i]["species"] for i in split["train_idx"]} == set(
            dataset.classes
        )
        assert split["train_origins"].isdisjoint(split["val_origins"])
        assert {dataset.records[i]["md5"] for i in split["train_idx"]}.isdisjoint(
            {dataset.records[i]["md5"] for i in split["val_idx"]}
        )

    import pandas as pd

    namespace = {
        "OriginKFoldDataset": cv.OriginKFoldDataset,
        "DATA_DIR": data,
        "ROOT": root,
        "CONFIG": {"n_folds": 5, "seed": 42},
        "pd": pd,
        "Path": Path,
    }
    notebook = json.loads((TASK_DIR / "fivefold_cv_training.ipynb").read_text())
    analysis = next(
        "".join(c["source"])
        for c in notebook["cells"]
        if "cv_splits = full_dataset" in "".join(c["source"])
    )
    exec(analysis, namespace)
    assert all(
        np.array_equal(a["val_idx"], b["val_idx"])
        for a, b in zip(splits, namespace["cv_splits"])
    )
