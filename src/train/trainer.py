"""Training engine for Conifer Family Classification.

Handles the training loop with mixed precision support and tqdm progress bars.
"""

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm


class TrainingEngine:
    """Training engine for Task A (Conifer Family Classification).

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        criterion: Loss function
        device: Device to use ('cuda' or 'cpu')
        use_amp: Use automatic mixed precision
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        criterion: Optional[nn.Module] = None,
        device: str = "cuda",
        use_amp: bool = True,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.device = device
        self.use_amp = use_amp
        self.scaler = GradScaler(self.device) if use_amp else None

        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.train_history = []
        self.val_history = []

    def train_one_epoch(
        self, epoch: int = 0, total_epochs: int = 1
    ) -> Dict[str, float]:
        """Train for one epoch with tqdm progress bar.

        Args:
            epoch: Current epoch number (for tqdm display)
            total_epochs: Total number of epochs (for tqdm display)

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{total_epochs} [Train]",
            unit="batch",
            leave=False,
        )
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast(self.device):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{100.0 * correct / total:.1f}%",
                }
            )

        pbar.close()

        metrics = {
            "train_loss": total_loss / len(self.train_loader),
            "train_acc": 100.0 * correct / total,
        }

        return metrics

    @torch.no_grad()
    def validate(self, epoch: int = 0, total_epochs: int = 1) -> Dict[str, float]:
        """Validate on validation set with tqdm progress bar.

        Args:
            epoch: Current epoch number (for tqdm display)
            total_epochs: Total number of epochs (for tqdm display)

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch+1}/{total_epochs} [Val  ]",
            unit="batch",
            leave=False,
        )
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(self.device, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{100.0 * correct / total:.1f}%",
                }
            )

        pbar.close()

        metrics = {
            "val_loss": total_loss / len(self.val_loader),
            "val_acc": 100.0 * correct / total,
        }

        return metrics

    def fit(
        self,
        epochs: int,
        output_dir: Path,
        name: str = "model",
        save_best_only: bool = True,
        early_stopping_patience: int = 0,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Train for multiple epochs with checkpointing and optional early stopping.

        Args:
            epochs: Number of epochs to train
            output_dir: Directory to save checkpoints
            name: Base name for checkpoint files
            save_best_only: Only save best model
            early_stopping_patience: Epochs to wait before stopping (0 = disabled)
            verbose: Print training progress

        Returns:
            Dictionary with final metrics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        early_stopping_counter = 0
        early_stopped = False

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_one_epoch(epoch, epochs)
            self.train_history.append(train_metrics)

            # Validate
            val_metrics = self.validate(epoch, epochs)
            self.val_history.append(val_metrics)

            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Update best accuracy and track best epoch
            is_best = val_metrics["val_acc"] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics["val_acc"]
                self.best_epoch = epoch
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            # Print progress with best epoch info
            if verbose:
                lr = self.optimizer.param_groups[0]["lr"]
                best_marker = "[Best]" if is_best else ""
                early_marker = f"[EarlyStop {early_stopping_counter}/{early_stopping_patience}]" if early_stopping_patience > 0 else ""
                print(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {train_metrics['train_loss']:.4f}, Acc: {train_metrics['train_acc']:.2f}% | "
                    f"Val Loss: {val_metrics['val_loss']:.4f}, Acc: {val_metrics['val_acc']:.2f}% | "
                    f"LR: {lr:.2e}{best_marker}{early_marker}"
                )

            # Save checkpoints
            if save_best_only:
                if is_best:
                    self.save_checkpoint(
                        output_dir / f"{name}_best.pth",
                        epoch=epoch,
                        metrics={**train_metrics, **val_metrics},
                    )
                    if verbose:
                        print(
                            f"  ↳ New best! Saved to {name}_best.pth (Val Acc: {val_metrics['val_acc']:.2f}%)"
                        )
            else:
                # Always save epoch checkpoint
                self.save_checkpoint(
                    output_dir / f"{name}_epoch{epoch+1}.pth",
                    epoch=epoch,
                    metrics={**train_metrics, **val_metrics},
                )
                if is_best:
                    self.save_checkpoint(
                        output_dir / f"{name}_best.pth",
                        epoch=epoch,
                        metrics={**train_metrics, **val_metrics},
                    )
                    if verbose:
                        print(
                            f"  ↳ New best! Saved to {name}_best.pth (Val Acc: {val_metrics['val_acc']:.2f}%)"
                        )

            # Early stopping check
            if early_stopping_patience > 0 and early_stopping_counter >= early_stopping_patience:
                print(f"\n[EarlyStopping] No improvement for {early_stopping_patience} epochs. Stopping at epoch {epoch+1}.")
                early_stopped = True
                break

        # Save last checkpoint
        self.save_checkpoint(
            output_dir / f"{name}_last.pth",
            epoch=self.current_epoch,
            metrics={**self.train_history[-1], **self.val_history[-1]},
        )

        return {
            "best_val_acc": self.best_val_acc,
            "best_epoch": self.best_epoch + 1,  # 1-indexed for user display
            "final_train_acc": self.train_history[-1]["train_acc"],
            "final_val_acc": self.val_history[-1]["val_acc"],
            "stopped_early": early_stopped,
            "epochs_trained": self.current_epoch + 1,
        }

    def save_checkpoint(self, path: Path, epoch: int, metrics: Dict):
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            metrics: Training metrics
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "best_val_acc": self.best_val_acc,
            "best_epoch": self.best_epoch,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)

    @staticmethod
    def load_checkpoint(
        path: Path, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint
            model: Model to load weights into
            optimizer: Optional optimizer to load state into

        Returns:
            Dictionary with checkpoint info
        """
        checkpoint = torch.load(path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return {
            "epoch": checkpoint.get("epoch", 0),
            "metrics": checkpoint.get("metrics", {}),
            "best_val_acc": checkpoint.get("best_val_acc", 0.0),
            "best_epoch": checkpoint.get("best_epoch", 0),
        }
