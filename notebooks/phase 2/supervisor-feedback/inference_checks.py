"""Deterministic inference probes shared by the Task 2 and Task 4 notebooks."""

from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import uuid

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import timm
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

from src.models.factory import BACKBONE_NAMES


ROBUSTNESS = ("baseline", "center_crop_70", "edge_mask_15", "grayscale")
DEGRADATION = ("baseline", "16x16", "blur_5", "blur_10")


def mask_border(image):
    border = round(image.width * 0.15)
    center = TF.crop(
        image, border, border, image.height - 2 * border, image.width - 2 * border
    )
    return TF.pad(center, border, fill=0)


def make_transform(condition):
    operations = [T.Resize((224, 224))]
    if condition == "center_crop_70":
        operations.extend([T.CenterCrop(round(224 * 0.7)), T.Resize((224, 224))])
    elif condition == "edge_mask_15":
        operations.append(T.Lambda(mask_border))
    elif condition == "grayscale":
        operations.append(T.Grayscale(num_output_channels=3))
    elif condition == "16x16":
        operations.extend([T.Resize((16, 16)), T.Resize((224, 224))])
    elif condition in ("blur_5", "blur_10"):
        sigma = float(condition.split("_")[1])
        operations.append(T.GaussianBlur(2 * round(3 * sigma) + 1, sigma=sigma))
    elif condition != "baseline":
        raise ValueError(f"Unknown inference probe: {condition}")
    operations.extend(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    return T.Compose(operations)


def evaluate_models(
    *, root, run_rows, checkpoint_paths, test_dir, conditions, output_parent,
    notebook, class_names, device, batch_size=32, source_registry=None,
    model_factory=None,
):
    """Evaluate saved weights sequentially; export one registry row per training run."""
    root, test_dir = Path(root).resolve(), Path(test_dir)
    dataset = ImageFolder(test_dir)
    if dataset.classes != list(class_names):
        raise ValueError("Test class order differs from the training class order")
    if not conditions or conditions[0] != "baseline" or len(set(conditions)) != len(conditions):
        raise ValueError("Conditions must begin with baseline and must be unique")
    if len({row["run_id"] for row in run_rows}) != len(run_rows):
        raise ValueError("Duplicate training run IDs")
    for row in run_rows:
        if len(dataset.classes) != int(row["num_classes"]):
            raise ValueError(f"Wrong class count for {row['run_id']}")
        if len(dataset) != int(row["n_test"]):
            raise ValueError(f"Wrong test sample count for {row['run_id']}")
        if not Path(checkpoint_paths[row["run_id"]]).is_file():
            raise FileNotFoundError(checkpoint_paths[row["run_id"]])

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(output_parent) / f"inference-{stamp}-{uuid.uuid4().hex[:8]}"
    (output / "predictions").mkdir(parents=True, exist_ok=False)
    registry_path = output / f"inference-{stamp}-registry.csv"
    results, registered = [], []
    create_model = timm.create_model if model_factory is None else model_factory
    device = torch.device(device)
    for source in run_rows:
        run_id, backbone = source["run_id"], source["backbone"]
        checkpoint_path = Path(checkpoint_paths[run_id]).resolve()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("class_to_idx", dataset.class_to_idx) != dataset.class_to_idx:
            raise ValueError(f"Checkpoint class mapping differs for {run_id}")
        model = create_model(
            BACKBONE_NAMES[backbone], pretrained=False, num_classes=len(class_names)
        )
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
        model.to(device).eval()
        epoch_raw = checkpoint.get("epoch")
        del checkpoint
        with checkpoint_path.open("rb") as handle:
            checkpoint_sha = hashlib.file_digest(handle, "sha256").hexdigest()
        baseline_accuracy = None
        for condition in conditions:
            dataset.transform = make_transform(condition)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            predictions, offset = [], 0
            with torch.inference_mode():
                for images, labels in tqdm(loader, desc=f"{backbone}: {condition}"):
                    predicted = model(images.to(device)).argmax(dim=1).cpu().tolist()
                    for label, prediction in zip(labels.tolist(), predicted):
                        path = Path(dataset.samples[offset][0]).resolve().relative_to(root).as_posix()
                        predictions.append(
                            dict(run_id=run_id, backbone=backbone, test=condition,
                                 path=path, true_label=label, predicted_label=prediction,
                                 true_species=class_names[label],
                                 predicted_species=class_names[prediction])
                        )
                        offset += 1
            prediction_path = output / "predictions" / f"{run_id}-{condition}.csv"
            pd.DataFrame(predictions).to_csv(prediction_path, index=False)
            y = [row["true_label"] for row in predictions]
            p = [row["predicted_label"] for row in predictions]
            accuracy = accuracy_score(y, p)
            macro_f1 = f1_score(y, p, labels=list(range(len(class_names))),
                                average="macro", zero_division=0)
            if condition == "baseline":
                baseline_accuracy = accuracy
                training_fields = (
                    "run_id", "run_date", "notebook", "git_commit", "split_version",
                    "num_classes", "n_train", "n_val", "n_test", "seed", "lr",
                    "epochs", "batch", "image_size", "label_smoothing", "augmentation",
                )
                registered.append(
                    dict(
                        **{key: source[key] for key in training_fields},
                        backbone=backbone, checkpoint_epoch_raw=epoch_raw,
                        checkpoint_path=checkpoint_path.relative_to(root).as_posix(),
                        checkpoint_sha256=checkpoint_sha,
                        prediction_path=prediction_path.resolve().relative_to(root).as_posix(),
                        test_accuracy=accuracy, test_f1_macro=macro_f1,
                    )
                )
            results.append(
                dict(run_id=run_id, baseline_run_id=run_id, backbone=backbone,
                     test=condition, accuracy=accuracy, macro_f1=macro_f1,
                     n_samples=len(predictions), drop_pp=(baseline_accuracy - accuracy) * 100,
                     prediction_path=prediction_path.resolve().relative_to(root).as_posix())
            )
            print(f"{backbone} | {condition}: accuracy {accuracy:.2%}, macro-F1 {macro_f1:.2%}")
        del model, loader, images
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results = pd.DataFrame(results)
    results.to_csv(output / "results.csv", index=False)
    pd.DataFrame(registered).to_csv(registry_path, index=False)
    (output / "metadata.json").write_text(
        json.dumps(
            dict(
                created_at_utc=stamp, registry=registry_path.resolve().relative_to(root).as_posix(),
                source_registry=source_registry, source_run_ids=[r["run_id"] for r in registered],
                evaluation_notebook=notebook, test_dir=test_dir.resolve().relative_to(root).as_posix(),
                class_to_idx=dataset.class_to_idx, evaluation_batch_size=batch_size,
                device=str(device), precision="float32; no autocast", conditions=list(conditions),
                preprocessing="Resize((224,224)) -> probe -> ToTensor -> ImageNet normalize",
                probe_parameters=dict(center_crop_ratio=0.7, edge_mask_ratio=0.15,
                                      grayscale_channels=3, degraded_size=16,
                                      blur_5=dict(sigma=5, kernel_size=31),
                                      blur_10=dict(sigma=10, kernel_size=61)),
                drop_unit="percentage points; own model baseline minus perturbed accuracy",
                training_registry_rule="one row per training run; all probes reuse its run_id",
                checkpoint_selection="recorded explicitly in registry checkpoint_path",
                inference_only=True,
            ), indent=2, ensure_ascii=False,
        ) + "\n", encoding="utf-8",
    )
    return results, output
