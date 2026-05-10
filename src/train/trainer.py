"""Training engine for Conifer Family Classification.

Handles the training loop with mixed precision support.
"""

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler


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
        self.scaler = GradScaler() if use_amp else None
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.train_history = []
        self.val_history = []
    
    def train_one_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
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
        
        metrics = {
            "train_loss": total_loss / len(self.train_loader),
            "train_acc": 100.0 * correct / total,
        }
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
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
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Train for multiple epochs with checkpointing.
        
        Args:
            epochs: Number of epochs to train
            output_dir: Directory to save checkpoints
            name: Base name for checkpoint files
            save_best_only: Only save best model
            verbose: Print training progress
        
        Returns:
            Dictionary with final metrics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_one_epoch()
            self.train_history.append(train_metrics)
            
            # Validate
            val_metrics = self.validate()
            self.val_history.append(val_metrics)
            
            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Update best accuracy
            if val_metrics["val_acc"] > self.best_val_acc:
                self.best_val_acc = val_metrics["val_acc"]
            
            # Print progress
            if verbose:
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {train_metrics['train_loss']:.4f}, Acc: {train_metrics['train_acc']:.2f}% | "
                    f"Val Loss: {val_metrics['val_loss']:.4f}, Acc: {val_metrics['val_acc']:.2f}% | "
                    f"LR: {lr:.2e}"
                )
            
            # Save checkpoints
            if save_best_only:
                if val_metrics["val_acc"] >= self.best_val_acc:
                    self.save_checkpoint(
                        output_dir / f"{name}_best.pth",
                        epoch=epoch,
                        metrics={**train_metrics, **val_metrics},
                    )
            else:
                self.save_checkpoint(
                    output_dir / f"{name}_epoch{epoch+1}.pth",
                    epoch=epoch,
                    metrics={**train_metrics, **val_metrics},
                )
        
        # Save last checkpoint
        self.save_checkpoint(
            output_dir / f"{name}_last.pth",
            epoch=epochs - 1,
            metrics={**self.train_history[-1], **self.val_history[-1]},
        )
        
        return {
            "best_val_acc": self.best_val_acc,
            "final_train_acc": self.train_history[-1]["train_acc"],
            "final_val_acc": self.val_history[-1]["val_acc"],
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
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
    
    @staticmethod
    def load_checkpoint(path: Path, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> Dict:
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
        }
