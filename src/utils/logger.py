"""Logging utilities for Conifer Family Classification."""

import logging
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "woodbench",
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Setup logger with file and console handlers.
    
    Args:
        name: Logger name
        log_dir: Directory for log file (None = only console)
        level: Logging level
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_dir provided)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"{name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class MetricsLogger:
    """Logger for tracking training metrics.
    
    Args:
        output_dir: Directory to save metrics
        name: Log file name
    """
    
    def __init__(self, output_dir: Path, name: str = "metrics"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / f"{name}.csv"
        
        # Create log file with header
        with open(self.log_file, "w") as f:
            f.write("epoch,train_loss,train_acc,val_loss,val_acc,lr\n")
    
    def log(self, epoch: int, train_metrics: dict, val_metrics: dict, lr: float):
        """Log metrics for an epoch.
        
        Args:
            epoch: Epoch number
            train_metrics: Training metrics dict
            val_metrics: Validation metrics dict
            lr: Learning rate
        """
        with open(self.log_file, "a") as f:
            f.write(
                f"{epoch},"
                f"{train_metrics.get('train_loss', 0):.6f},"
                f"{train_metrics.get('train_acc', 0):.4f},"
                f"{val_metrics.get('val_loss', 0):.6f},"
                f"{val_metrics.get('val_acc', 0):.4f},"
                f"{lr:.2e}\n"
            )
