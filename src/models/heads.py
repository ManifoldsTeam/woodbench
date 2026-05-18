"""Classification heads for Conifer Family Classification."""

import torch.nn as nn


class ClassificationHead(nn.Module):
    """Simple linear classification head."""
    
    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        layers = [nn.Linear(in_features, num_classes)]
        if dropout > 0:
            layers.insert(0, nn.Dropout(dropout))
        self.head = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.head(x)


class DropoutHead(nn.Module):
    """Classification head with dropout for regularization."""
    
    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        x = self.dropout(x)
        return self.fc(x)
