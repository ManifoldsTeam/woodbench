"""Confusion matrix visualization for Conifer Family Classification."""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    normalize: bool = False,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """Plot confusion matrix.
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        normalize: Whether to normalize values
        title: Plot title
        cmap: Colormap name
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = "d"
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
    )
    
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    return fig


def save_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path,
    normalize: bool = False,
    title: str = "Confusion Matrix",
    dpi: int = 300,
) -> None:
    """Plot and save confusion matrix.
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        output_path: Path to save figure
        normalize: Whether to normalize values
        title: Plot title
        dpi: Resolution for saved figure
    """
    fig = plot_confusion_matrix(cm, class_names, normalize, title)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_f1(
    per_class_metrics: dict,
    class_names: List[str],
    output_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Plot per-class F1 scores as a bar chart.
    
    Args:
        per_class_metrics: Dictionary with per-class metrics
        class_names: List of class names
        output_path: Optional path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    f1_scores = [per_class_metrics[name]["f1"] for name in class_names]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(class_names)))
    bars = ax.bar(class_names, f1_scores, color=colors)
    
    # Add value labels on bars
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        ax.annotate(
            f"{score:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    
    ax.set_xlabel("Family")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Scores")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.95, color="r", linestyle="--", alpha=0.5, label="Target (0.95)")
    ax.axhline(y=1.0, color="g", linestyle="--", alpha=0.5, label="Perfect (1.0)")
    ax.legend()
    
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    
    return fig
