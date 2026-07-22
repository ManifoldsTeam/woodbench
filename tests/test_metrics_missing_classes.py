import torch
from torch.utils.data import DataLoader, TensorDataset

from src.eval.metrics import compute_metrics


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
