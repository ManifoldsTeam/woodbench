"""Inference probes must share the paper's full-image evaluation pipeline."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms as T


ROOT = Path(__file__).resolve().parents[1]
FEEDBACK = ROOT / "notebooks/phase 2/supervisor-feedback"
HELPER = FEEDBACK / "inference_checks.py"


def _transform(condition):
    return _helper().make_transform(condition)


def _wide_image():
    pixels = np.zeros((100, 300, 3), dtype=np.uint8)
    pixels[:, :70] = (255, 0, 0)
    pixels[:, 70:230] = (0, 255, 0)
    pixels[:, 230:] = (0, 0, 255)
    return Image.fromarray(pixels)


def _normalize(image):
    return T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )(image)


def test_degradation_baseline_keeps_entire_image_as_in_paper():
    image = _wide_image()
    expected = _normalize(T.Resize((224, 224))(image))
    torch.testing.assert_close(_transform("baseline")(image), expected)


@pytest.mark.parametrize("sigma,kernel", [(5, 31), (10, 61)])
def test_blur_changes_only_sharpness_after_common_resize(sigma, kernel):
    image = _wide_image()
    resized = T.Resize((224, 224))(image)
    expected = _normalize(T.GaussianBlur(kernel, sigma=float(sigma))(resized))
    torch.testing.assert_close(_transform(f"blur_{sigma}")(image), expected)


def test_16px_changes_resolution_after_common_resize():
    image = _wide_image()
    resized = T.Resize((224, 224))(image)
    small = T.Resize((16, 16))(resized)
    expected = _normalize(T.Resize((224, 224))(small))
    torch.testing.assert_close(_transform("16x16")(image), expected)


def _helper():
    spec = importlib.util.spec_from_file_location("inference_checks", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("condition", ["center_crop_70", "edge_mask_15", "grayscale"])
def test_robustness_probes_are_deterministic_rgb_tensors(condition):
    transform = _transform(condition)
    image = _wide_image()
    first = transform(image)
    assert first.shape == (3, 224, 224)
    torch.testing.assert_close(first, transform(image))


class _Classifier(torch.nn.Module):
    def __init__(self, predicted_label):
        super().__init__()
        self.register_buffer("predicted_label", torch.tensor(predicted_label))

    def forward(self, images):
        scores = torch.zeros((len(images), 2), device=images.device)
        scores[:, int(self.predicted_label)] = 1
        return scores


def _runs(tmp_path):
    rows, checkpoints = [], {}
    for label, count in [("A", 2), ("B", 1)]:
        folder = tmp_path / "test" / label
        folder.mkdir(parents=True)
        for i in range(count):
            _wide_image().save(folder / f"{i}.png")
    for index, backbone in enumerate(["convnext_tiny", "mobilenetv3_large"]):
        run_id = f"paper-{backbone}"
        rows.append(
            dict(
                run_id=run_id, backbone=backbone, num_classes=2, n_test=3,
                n_train=6, n_val=2, run_date="2026-08-19", notebook="train.ipynb",
                git_commit="source-commit", split_version="test-split", seed=42,
                lr=0.0001, epochs=50, batch=32, image_size=224,
                label_smoothing=0.1, augmentation="main",
            )
        )
        path = tmp_path / f"{backbone}.pth"
        torch.save({"model_state_dict": _Classifier(index).state_dict()}, path)
        checkpoints[run_id] = path
    return rows, checkpoints


def test_two_models_use_own_baselines_and_export_traceable_results(tmp_path):
    helper = _helper()
    rows, checkpoints = _runs(tmp_path)
    results, output = helper.evaluate_models(
        root=tmp_path, run_rows=rows, checkpoint_paths=checkpoints,
        test_dir=tmp_path / "test", conditions=helper.ROBUSTNESS,
        output_parent=tmp_path / "runs", notebook="evaluate.ipynb",
        class_names=["A", "B"], device="cpu", batch_size=2,
        model_factory=lambda name, **kwargs: _Classifier(0),
    )
    assert len(results) == 8
    baseline = results[results["test"] == "baseline"].set_index("backbone")
    assert baseline.loc["convnext_tiny", "accuracy"] == pytest.approx(2 / 3)
    assert baseline.loc["mobilenetv3_large", "accuracy"] == pytest.approx(1 / 3)
    assert results["drop_pp"].eq(0).all()
    metadata = json.loads((output / "metadata.json").read_text())
    import pandas as pd
    registry = pd.read_csv(tmp_path / metadata["registry"])
    assert len(registry) == 2
    assert set(results["run_id"]) == set(registry["run_id"])
    assert (results["run_id"] == results["baseline_run_id"]).all()
    for row in registry.to_dict("records"):
        prediction = pd.read_csv(tmp_path / row["prediction_path"])
        assert len(prediction) == 3
        assert prediction["test"].eq("baseline").all()
        assert prediction["run_id"].eq(row["run_id"]).all()


def test_inference_rejects_mismatched_class_order_before_loading_models(tmp_path):
    helper = _helper()
    rows, checkpoints = _runs(tmp_path)
    with pytest.raises(ValueError, match="class"):
        helper.evaluate_models(
            root=tmp_path, run_rows=rows, checkpoint_paths=checkpoints,
            test_dir=tmp_path / "test", conditions=helper.DEGRADATION,
            output_parent=tmp_path / "runs", notebook="evaluate.ipynb",
            class_names=["B", "A"], device="cpu", batch_size=2,
        )


def test_friendly_mobile_name_resolves_to_real_timm_architecture(tmp_path):
    helper = _helper()
    rows, checkpoints = _runs(tmp_path)
    names = []

    def create_model(name, **kwargs):
        names.append(name)
        return _Classifier(0)

    helper.evaluate_models(
        root=tmp_path, run_rows=rows, checkpoint_paths=checkpoints,
        test_dir=tmp_path / "test", conditions=["baseline"],
        output_parent=tmp_path / "runs", notebook="evaluate.ipynb",
        class_names=["A", "B"], device="cpu", model_factory=create_model,
    )
    assert names == ["convnext_tiny", "mobilenetv3_large_100"]


@pytest.mark.parametrize(
    "folder,filename,conditions",
    [
        ("task_02_robustness_tests", "robustness_tests.ipynb",
         {"baseline", "center_crop_70", "edge_mask_15", "grayscale"}),
        ("task_04_degradation_rerun", "degradation_rerun.ipynb",
         {"baseline", "16x16", "blur_5", "blur_10"}),
    ],
)
def test_notebook_runs_all_cells_on_small_fixture(
    tmp_path, monkeypatch, folder, filename, conditions
):
    import pandas as pd
    import timm
    import matplotlib
    import nbformat

    matplotlib.use("Agg")
    rows, checkpoints = _runs(tmp_path)
    dataset = tmp_path / "data/task-B-group-split/test"
    dataset.parent.mkdir(parents=True)
    (tmp_path / "test").rename(dataset)
    checkpoint_root = tmp_path / "data/experiments/Task-B-new/19-08-2026/main/seed_42"
    for row in rows:
        path = checkpoint_root / row["backbone"] / f"{row['backbone']}_seed42_best.pth"
        path.parent.mkdir(parents=True)
        path.write_bytes(checkpoints[row["run_id"]].read_bytes())
        row["run_id"] = f"taskB17__main__seed_42__{row['backbone']}"
    source = tmp_path / "docs/reports/experiments/phase_1b/final_b_new_recomputed/20260819T043733Z"
    source.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(source / "task-b-17class-20260819T043733Z-registry.csv", index=False)
    (source / "metadata.json").write_text(json.dumps({"classes": ["A", "B"]}))
    created = []

    def create_model(name, **kwargs):
        created.append(name)
        return _Classifier(0)

    monkeypatch.setattr(timm, "create_model", create_model)
    monkeypatch.syspath_prepend(str(FEEDBACK))
    notebook = nbformat.read(FEEDBACK / folder / filename, as_version=4)
    nbformat.validate(notebook)
    namespace = {}
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        assert cell.execution_count is None and not cell.outputs
        exec(compile(cell.source, f"{filename}:cell", "exec"), namespace)
        # Execute real root-detection first, then use isolated dataset/checkpoints.
        if "run_rows" not in namespace:
            namespace["ROOT"] = tmp_path
            namespace["FEEDBACK"] = tmp_path / "notebooks/phase 2/supervisor-feedback"
    assert created == ["convnext_tiny", "mobilenetv3_large_100"]
    assert namespace["CHECKPOINT_KIND"] == "best"
    assert set(namespace["results_df"]["test"]) == conditions
    assert len(namespace["results_df"]) == 8
    output = namespace["OUTPUT_DIR"]
    assert (output / "comparison.png").exists()
    assert json.loads((output / "metadata.json").read_text())["inference_only"]
