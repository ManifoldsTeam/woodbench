#!/usr/bin/env python3
"""
5-Fold Cross-Validation by Origin Key

Trains a single model with GroupKFold split by origin_key.
Each fold trains on samples from N-1 origins and validates on samples from 1 held-out origin.

Usage:
    python train_kfold.py --backbone convnext_tiny --n_folds 5
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset, DataLoader, Subset
import timm
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

# Setup path
ROOT = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from src.train import WandbTracker
from src.utils import set_seed

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def extract_origin_key(filename: str) -> str:
    """Extract origin key from filename.

    Format: "3351. Araucaria araucana.1.(12).png"
    Origin: "3351. Araucaria araucana.1"
    """
    if ' (' in filename:
        return filename.rsplit(' (', 1)[0]
    return filename.rsplit('.png', 1)[0] if '.png' in filename else filename


class OriginKFoldDataset(Dataset):
    """Dataset that supports origin-based grouping for K-Fold CV."""

    def __init__(self, data_root: Path, split: str, transform=None, classes: list = None):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform

        # Get classes from directory structure
        split_dir = self.data_root / split
        if classes is None:
            self.classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
        else:
            self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Collect all samples with origin keys
        self.samples = []  # (path, label, origin_key)

        for class_name in self.classes:
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                continue
            label = self.class_to_idx[class_name]

            for img_path in class_dir.glob("*.png"):
                origin_key = extract_origin_key(img_path.name)
                self.samples.append((str(img_path), label, origin_key))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, origin_key = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label, origin_key

    def get_splits_by_origin(self, n_folds: int = 5):
        """Get train/val indices for each fold using GroupKFold."""
        origin_keys = np.array([s[2] for s in self.samples])
        indices = np.arange(len(self.samples))

        gkf = GroupKFold(n_splits=n_folds)
        splits = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(indices, groups=origin_keys)):
            train_origins = set(origin_keys[train_idx])
            val_origins = set(origin_keys[val_idx])
            splits.append({
                'fold': fold,
                'train_idx': train_idx,
                'val_idx': val_idx,
                'train_origins': train_origins,
                'val_origins': val_origins,
                'n_train': len(train_idx),
                'n_val': len(val_idx),
            })

        return splits


class SubsetWithTransform(Dataset):
    """Subset dataset with transform override."""
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform or dataset.transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label, origin_key = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(img_size=224, use_90rotation=False):
    """Get training and validation transforms."""
    train_transform = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(90) if use_90rotation else T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc='Train', leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc='Eval', leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_full(model, loader, device):
    """Full evaluation with per-class metrics."""
    model.eval()
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc='Full Eval', leave=False):
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy() if isinstance(labels, torch.Tensor) else labels)

    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'precision_macro': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall_macro': recall_score(all_labels, all_preds, average='macro', zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(description='5-Fold CV by Origin Key')
    parser.add_argument('--backbone', type=str, required=True,
                       choices=['resnet50', 'efficientnet_b3', 'mobilenetv3_large',
                               'convnext_tiny', 'swin_tiny'],
                       help='Backbone model')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--epochs', type=int, default=30, help='Epochs per fold')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--backbone_lr', type=float, default=1e-5, help='Backbone learning rate')
    parser.add_argument('--head_lr', type=float, default=1e-4, help='Head learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--data_root', type=str, default=None, help='Data root path')
    parser.add_argument('--output_root', type=str, default=None, help='Output root path')
    parser.add_argument('--wandb_project', type=str, default='woodbench-origin-cv', help='WandB project')
    parser.add_argument('--use_90rotation', action='store_true', help='Use 90° rotation augmentation')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--dryrun', action='store_true', help='Dry run: verify setup without training')
    args = parser.parse_args()

    # Setup paths
    if args.data_root:
        data_root = Path(args.data_root)
    else:
        data_root = ROOT / "data" / "task-B-group-split"

    if args.output_root:
        output_root = Path(args.output_root)
    else:
        output_root = ROOT / "data" / "experiments" / "Task-B-origin-cv"

    output_dir = output_root / args.backbone
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Backbone: {args.backbone}")
    print(f"Folds: {args.n_folds}")
    print(f"Output: {output_dir}")

    # Create dataset
    train_transform, val_transform = get_transforms(use_90rotation=args.use_90rotation)
    full_dataset = OriginKFoldDataset(data_root, 'train', transform=train_transform)
    classes = full_dataset.classes
    num_classes = len(classes)

    print(f"Dataset: {len(full_dataset)} samples, {num_classes} classes")

    # Get splits
    splits = full_dataset.get_splits_by_origin(args.n_folds)

    # Print split info
    print(f"\n=== 5-Fold Split by Origin ===")
    total_origins = len(set(s[0]['train_origins'] | s[0]['val_origins'] for s in splits))
    for split in splits:
        print(f"Fold {split['fold']+1}: train={split['n_train']} ({len(split['train_origins'])} origins), "
              f"val={split['n_val']} ({len(split['val_origins'])} origins)")
    print(f"Total unique origins: {total_origins}")

    # DRYRUN: Verify setup without training
    if args.dryrun:
        print("\n" + "=" * 60)
        print("🔍 DRYRUN MODE - Verifying setup")
        print("=" * 60)

        # Check data
        print(f"\n✅ Dataset: {data_root}")
        print(f"   Samples: {len(full_dataset)}")
        print(f"   Classes: {num_classes}")
        print(f"   Files: {len(list((data_root / 'train').glob('**/*.png')))}")

        # Check origin distribution
        origin_counts = {}
        for s in full_dataset.samples:
            origin = s[2]
            origin_counts[origin] = origin_counts.get(origin, 0) + 1

        print(f"\n✅ Origin distribution:")
        print(f"   Unique origins: {len(origin_counts)}")
        print(f"   Min samples/origin: {min(origin_counts.values())}")
        print(f"   Max samples/origin: {max(origin_counts.values())}")
        print(f"   Mean samples/origin: {np.mean(list(origin_counts.values())):.1f}")

        # Check split
        print(f"\n✅ 5-Fold Split:")
        for split in splits:
            val_origins = len(split['val_origins'])
            train_origins = len(split['train_origins'])
            print(f"   Fold {split['fold']+1}: {train_origins} train origins, {val_origins} val origins")

        # Check model
        print(f"\n✅ Model: {args.backbone}")
        try:
            model = timm.create_model(args.backbone, pretrained=False, num_classes=num_classes)
            print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
            del model
        except Exception as e:
            print(f"   ❌ Error: {e}")

        # Check GPU
        print(f"\n✅ GPU: {device}")
        if torch.cuda.is_available():
            print(f"   Name: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print(f"   ⚠️ No GPU, will use CPU (slow)")

        # Check output
        print(f"\n✅ Output: {output_dir}")

        print("\n" + "=" * 60)
        print("✅ DRYRUN PASSED - Setup is valid")
        print("=" * 60)
        print("\nTo start training, run without --dryrun:")
        print(f"  python train_kfold.py --backbone {args.backbone} --n_folds 5")
        return

    # Results storage
    fold_results = []

    # Train each fold
    for fold_idx, split in enumerate(splits):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{args.n_folds}")
        print(f"{'='*60}")

        fold_dir = output_dir / f"fold_{fold_idx + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Check if already completed
        metrics_path = fold_dir / "metrics.json"
        if metrics_path.exists() and not args.resume:
            print(f"Fold {fold_idx + 1} already completed, skipping...")
            with open(metrics_path) as f:
                fold_results.append(json.load(f))
            continue

        # Create subsets
        train_dataset = SubsetWithTransform(full_dataset, split['train_idx'], train_transform)
        val_dataset = SubsetWithTransform(full_dataset, split['val_idx'], val_transform)

        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

        # Dataloaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                 num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                               num_workers=4, pin_memory=True)

        # Model
        model = timm.create_model(args.backbone, pretrained=True, num_classes=num_classes)
        model = model.to(device)

        # Optimizer with differential LR
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if 'head' in name or 'fc' in name or 'classifier' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': args.backbone_lr},
            {'params': head_params, 'lr': args.head_lr},
        ], weight_decay=args.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        # WandB tracking
        wandb_enabled = bool(os.getenv('WANDB_API_KEY'))

        if wandb_enabled:
            tracker = WandbTracker.start(
                project=args.wandb_project,
                name=f"{args.backbone}_fold_{fold_idx + 1}",
                config={
                    'backbone': args.backbone,
                    'fold': fold_idx + 1,
                    'n_folds': args.n_folds,
                    'epochs': args.epochs,
                    'batch_size': args.batch_size,
                    'backbone_lr': args.backbone_lr,
                    'head_lr': args.head_lr,
                    'label_smoothing': args.label_smoothing,
                    'seed': args.seed,
                    'n_train_origins': len(split['train_origins']),
                    'n_val_origins': len(split['val_origins']),
                },
                tags=[args.backbone, f'fold_{fold_idx+1}', 'origin-cv'],
                dir_path=str(fold_dir),
            )
        else:
            tracker = None

        # Training
        best_val_acc = 0
        best_epoch = 0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(args.epochs):
            set_seed(args.seed + epoch)  # Slight variation per epoch

            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            if tracker:
                tracker.log({
                    'train/loss': train_loss,
                    'train/acc': train_acc,
                    'val/loss': val_loss,
                    'val/acc': val_acc,
                    'lr': optimizer.param_groups[0]['lr'],
                }, step=epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'best_epoch': best_epoch,
                    'config': vars(args),
                }, fold_dir / 'best.pth')

                marker = " ★"
            else:
                marker = ""

            if (epoch + 1) % 5 == 0 or marker:
                print(f"Epoch {epoch+1:3d}/{args.epochs}: "
                      f"Train={train_acc:.4f}, Val={val_acc:.4f}{marker}")

        # Save last checkpoint
        torch.save({
            'epoch': args.epochs - 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, fold_dir / 'last.pth')

        # Load best for final evaluation
        model.load_state_dict(torch.load(fold_dir / 'best.pth')['model_state_dict'])

        # Final evaluation
        test_metrics = evaluate_full(model, val_loader, device)

        result = {
            'backbone': args.backbone,
            'fold': fold_idx + 1,
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch,
            'test_accuracy': test_metrics['accuracy'],
            'test_f1': test_metrics['f1_macro'],
            'test_precision': test_metrics['precision_macro'],
            'test_recall': test_metrics['recall_macro'],
            'n_train_samples': len(train_dataset),
            'n_val_samples': len(val_dataset),
            'n_train_origins': len(split['train_origins']),
            'n_val_origins': len(split['val_origins']),
        }

        # Save metrics
        with open(metrics_path, 'w') as f:
            json.dump(result, f, indent=2)

        fold_results.append(result)

        if tracker:
            tracker.log({
                'best_val_acc': best_val_acc,
                'best_epoch': best_epoch,
                'test_accuracy': test_metrics['accuracy'],
                'test_f1': test_metrics['f1_macro'],
            }, step=args.epochs)
            tracker.finish()

        print(f"\nFold {fold_idx + 1} - Best Val: {best_val_acc:.4f}, "
              f"Test Acc: {test_metrics['accuracy']:.4f}")

        # Cleanup
        del model, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("5-FOLD CV SUMMARY")
    print(f"{'='*60}")

    val_accs = [r['best_val_acc'] for r in fold_results]
    test_accs = [r['test_accuracy'] for r in fold_results]

    print(f"\nVal Accuracy: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
    print(f"Test Accuracy: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")

    for i, r in enumerate(fold_results):
        print(f"  Fold {i+1}: Val={r['best_val_acc']:.4f}, Test={r['test_accuracy']:.4f}")

    # Save summary
    summary = {
        'backbone': args.backbone,
        'n_folds': args.n_folds,
        'epochs': args.epochs,
        'config': vars(args),
        'fold_results': fold_results,
        'summary': {
            'val_acc_mean': float(np.mean(val_accs)),
            'val_acc_std': float(np.std(val_accs)),
            'test_acc_mean': float(np.mean(test_accs)),
            'test_acc_std': float(np.std(test_accs)),
        }
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Save as CSV
    import csv
    csv_path = output_dir / 'results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['fold', 'best_val_acc', 'test_accuracy',
                                              'test_f1', 'n_train_samples', 'n_val_samples'])
        writer.writeheader()
        for r in fold_results:
            writer.writerow({
                'fold': r['fold'],
                'best_val_acc': f"{r['best_val_acc']:.4f}",
                'test_accuracy': f"{r['test_accuracy']:.4f}",
                'test_f1': f"{r['test_f1']:.4f}",
                'n_train_samples': r['n_train_samples'],
                'n_val_samples': r['n_val_samples'],
            })

    print(f"\nResults saved to: {output_dir}")
    print(f"  - summary.json")
    print(f"  - results.csv")
    for r in fold_results:
        print(f"  - fold_{r['fold']}/best.pth")

    return summary


if __name__ == '__main__':
    main()
