"""Root module for Conifer Family Classification."""
from .data import ConiferDataset, get_dataloaders, get_train_transforms, get_val_transforms
from .models import create_model, BACKBONE_NAMES
from .train import (
    TrainingEngine,
    create_optimizer,
    create_scheduler,
    create_loss_function,
)
from .eval import compute_metrics, measure_inference_time, plot_confusion_matrix, save_confusion_matrix
from .utils import set_seed, setup_logger

__all__ = [
    # Data
    "ConiferDataset",
    "get_dataloaders",
    "get_train_transforms",
    "get_val_transforms",
    # Models
    "create_model",
    "BACKBONE_NAMES",
    # Training
    "TrainingEngine",
    "create_optimizer",
    "create_scheduler",
    "create_loss_function",
    # Evaluation
    "compute_metrics",
    "measure_inference_time",
    "plot_confusion_matrix",
    "save_confusion_matrix",
    # Utils
    "set_seed",
    "setup_logger",
]
