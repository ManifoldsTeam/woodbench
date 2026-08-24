import torch
from torch.utils.data import DataLoader, TensorDataset

import pytest

from src.eval.metrics import (
    TASK_B_LABELS,
    TASK_B_NUM_CLASSES,
    compute_metrics,
    compute_task_b_metrics,
)


class FixedLogitModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", torch.tensor(logits, dtype=torch.float32))

    def forward(self, images):
        return self.logits[: images.shape[0]]


def test_compute_metrics_keeps_missing_classes_in_outputs():
    images = torch.zeros(2, 3, 4, 4)
    labels = torch.tensor([0, 2])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    model = FixedLogitModel(
        [
            [10.0, 0.0, 0.0],
            [0.0, 0.0, 10.0],
        ]
    )

    metrics = compute_metrics(
        model=model,
        data_loader=loader,
        device="cpu",
        num_classes=3,
        class_names=["A", "B", "C"],
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["confusion_matrix"].shape == (3, 3)
    assert set(metrics["per_class"]) == {"A", "B", "C"}
    assert metrics["per_class"]["A"]["f1"] == 1.0
    assert metrics["per_class"]["B"]["f1"] == 0.0
    assert metrics["per_class"]["C"]["f1"] == 1.0
    assert metrics["f1_macro"] == pytest.approx(2 / 3)
    assert metrics["labels"] == [0, 1, 2]
    assert metrics["average"] == "macro"
    assert metrics["zero_division"] == 0


def test_task_b_metrics_always_use_all_17_labels_when_classes_are_missing():
    class_names = [f"species_{label}" for label in TASK_B_LABELS]
    metrics = compute_task_b_metrics(
        y_true=[0, 2],
        y_pred=[0, 2],
        class_names=class_names,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["f1_macro"] == pytest.approx(2 / TASK_B_NUM_CLASSES)
    assert metrics["precision_macro"] == pytest.approx(2 / TASK_B_NUM_CLASSES)
    assert metrics["recall_macro"] == pytest.approx(2 / TASK_B_NUM_CLASSES)
    assert metrics["confusion_matrix"].shape == (
        TASK_B_NUM_CLASSES,
        TASK_B_NUM_CLASSES,
    )
    assert metrics["per_class"]["species_1"] == {
        "label": 1,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }


def test_task_b_metrics_cover_all_final_b_labels():
    labels = list(TASK_B_LABELS)
    class_names = [f"species_{label}" for label in labels]
    metrics = compute_task_b_metrics(
        y_true=labels,
        y_pred=labels,
        class_names=class_names,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["f1_macro"] == 1.0
    assert metrics["labels"] == labels
    assert metrics["class_names"] == class_names


def test_task_b_metrics_reject_predictions_outside_fixed_labels():
    with pytest.raises(ValueError, match="outside labels"):
        compute_task_b_metrics(
            y_true=[0, 1],
            y_pred=[0, TASK_B_NUM_CLASSES],
        )


def test_task_b_metrics_reject_non_17_class_names():
    with pytest.raises(ValueError, match="same length as labels"):
        compute_task_b_metrics(
            y_true=[0, 1],
            y_pred=[0, 1],
            class_names=["A", "B"],
        )
