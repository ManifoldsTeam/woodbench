#!/usr/bin/env python3
"""
Train a single fold for 5-Fold CV by Origin Key.

This is the entry point for running one fold at a time (useful for debugging).

Usage:
    python train_single_fold.py --backbone convnext_tiny --fold 1
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset, DataLoader
import timm
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

# Setup path
ROOT = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from src.utils import set_seed


def extract_origin_key(filename: str) -> str:
    """Extract origin key from filename."""
    if ' (' in filename:
        return filename.rsplit(' (', 1)[0]
    return filename.rsplit('.png', 1)[0] if '.png' in filename else filename


class OriginDataset(Dataset):
    """Dataset that supports origin-based grouping."""

    def __init__(self, data_root: Path, split: str, transform=None):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform

        split_dir = self.data_root / split
        self.classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

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


class SubsetWithTransform(Dataset):
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


def get_transforms(use_90rotation=False):
    train_transform = T.Compose([
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(90) if use_90rotation else T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

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
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc='Eval', leave=False):
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy() if isinstance(labels, torch.Tensor) else labels)

    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='macro', zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--backbone_lr', type=float, default=1e-5)
    parser.add_argument('--head_lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--output_root', type=str, default=None)
    parser.add_argument('--use_90rotation', action='store_true')
    args = parser.parse_args()

    # Paths
    data_root = Path(args.data_root) if args.data_root else ROOT / "data" / "task-B-group-split"
    output_root = Path(args.output_root) if args.output_root else ROOT / "data" / "experiments" / "Task-B-origin-cv"

    output_dir = output_root / args.backbone / f"fold_{args.fold}"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Backbone: {args.backbone}, Fold: {args.fold}")

    # Dataset and splits
    train_transform, val_transform = get_transforms(args.use_90rotation)
    dataset = OriginDataset(data_root, 'train', transform=train_transform)

    origin_keys = np.array([s[2] for s in dataset.samples])
    indices = np.arange(len(dataset))

    gkf = GroupKFold(n_splits=args.n_folds)
    splits = list(gkf.split(indices, groups=origin_keys))

    train_idx, val_idx = splits[args.fold - 1]

    train_dataset = SubsetWithTransform(dataset, train_idx, train_transform)
    val_dataset = SubsetWithTransform(dataset, val_idx, val_transform)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                          num_workers=4, pin_memory=True)

    # Model
    model = timm.create_model(args.backbone, pretrained=True, num_classes=len(dataset.classes))
    model = model.to(device)

    # Optimizer
    backbone_params = [p for n, p in model.named_parameters() if 'head' not in n and 'fc' not in n]
    head_params = [p for n, p in model.named_parameters() if 'head' in n or 'fc' in n]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.backbone_lr},
        {'params': head_params, 'lr': args.head_lr},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Training
    best_val_acc = 0
    best_epoch = 0

    for epoch in range(args.epochs):
        set_seed(args.seed + epoch)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: Train={train_acc:.4f}")

    # Evaluate
    metrics = evaluate(model, val_loader, device)

    # Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': metrics['accuracy'],
        'epoch': args.epochs - 1,
    }, output_dir / 'best.pth')

    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump({
            'backbone': args.backbone,
            'fold': args.fold,
            'accuracy': metrics['accuracy'],
            'f1': metrics['f1'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'n_train': len(train_dataset),
            'n_val': len(val_dataset),
        }, f, indent=2)

    print(f"Fold {args.fold} complete: Acc={metrics['accuracy']:.4f}")

    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
