from pathlib import Path

from src.data.dataset import FlatConiferDataset, get_dataloaders


def _touch_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not-used-by-this-test")


def test_flat_dataset_can_use_fixed_class_mapping(tmp_path):
    data_root = tmp_path / "data"
    _touch_png(data_root / "train" / "A" / "a.png")
    _touch_png(data_root / "train" / "B" / "b.png")
    _touch_png(data_root / "train" / "C" / "c.png")
    _touch_png(data_root / "val" / "A" / "a_val.png")
    _touch_png(data_root / "val" / "C" / "c_val.png")

    train = FlatConiferDataset(data_root, split="train")
    val = FlatConiferDataset(data_root, split="val", classes=train.classes)

    assert train.classes == ["A", "B", "C"]
    assert val.classes == train.classes
    assert val.class_to_idx == {"A": 0, "B": 1, "C": 2}
    assert sorted(label for _, label in val.samples) == [0, 2]


def test_get_dataloaders_uses_train_class_mapping_for_eval_splits(tmp_path):
    data_root = tmp_path / "data"
    _touch_png(data_root / "train" / "A" / "a.png")
    _touch_png(data_root / "train" / "B" / "b.png")
    _touch_png(data_root / "train" / "C" / "c.png")
    _touch_png(data_root / "val" / "A" / "a_val.png")
    _touch_png(data_root / "val" / "C" / "c_val.png")
    _touch_png(data_root / "test" / "B" / "b_test.png")
    _touch_png(data_root / "test" / "C" / "c_test.png")

    train_loader, val_loader, test_loader = get_dataloaders(
        data_root,
        batch_size=2,
        num_workers=0,
        task="taskB",
    )

    assert train_loader.dataset.classes == ["A", "B", "C"]
    assert val_loader.dataset.classes == train_loader.dataset.classes
    assert test_loader.dataset.classes == train_loader.dataset.classes
    assert sorted(label for _, label in val_loader.dataset.samples) == [0, 2]
    assert sorted(label for _, label in test_loader.dataset.samples) == [1, 2]
