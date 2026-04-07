"""
Visualization utilities for skeleton and training — v2
Enhanced from BSQAv1 with attention weight visualization support.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
from ..data.skeleton import SKELETON_EDGES, KEYPOINT_NAMES


def plot_skeleton(
    keypoints: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "",
    show_labels: bool = False,
) -> plt.Axes:
    """
    Plot a single skeleton frame.

    Args:
        keypoints: (17, 2) array of keypoint coordinates
        ax: Matplotlib axes (created if None)
        title: Plot title
        show_labels: Whether to show keypoint labels

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    for start, end in SKELETON_EDGES:
        x = [keypoints[start, 0], keypoints[end, 0]]
        y = [keypoints[start, 1], keypoints[end, 1]]
        if np.allclose(keypoints[start], 0) or np.allclose(keypoints[end], 0):
            continue
        ax.plot(x, y, 'b-', linewidth=2, alpha=0.7)

    valid_mask = ~(np.isclose(keypoints[:, 0], 0) & np.isclose(keypoints[:, 1], 0))
    ax.scatter(
        keypoints[valid_mask, 0],
        keypoints[valid_mask, 1],
        c='red', s=50, zorder=5
    )

    if show_labels:
        for i, (x, y) in enumerate(keypoints):
            if not np.isclose(x, 0) or not np.isclose(y, 0):
                ax.annotate(KEYPOINT_NAMES[i], (x, y), fontsize=8)

    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title(title)

    return ax


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training loss and accuracy curves.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, 'b-', label='Train')
    ax1.plot(epochs, val_losses, 'r-', label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, 'b-', label='Train')
    ax2.plot(epochs, val_accs, 'r-', label='Val')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_attention_weights(
    attention_weights: np.ndarray,
    frame_labels: Optional[List[str]] = None,
    title: str = "Temporal Attention Weights",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize temporal attention weights over frames.
    Shows which frames the model considers most important.

    Args:
        attention_weights: (seq_len,) array of attention scores
        frame_labels: Optional labels for each frame
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    seq_len = len(attention_weights)
    fig, ax = plt.subplots(figsize=(14, 3))

    frames = np.arange(seq_len)
    colors = plt.cm.YlOrRd(attention_weights / attention_weights.max())

    ax.bar(frames, attention_weights, color=colors, width=1.0, edgecolor='none')
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Attention Weight')
    ax.set_title(title)
    ax.set_xlim(-0.5, seq_len - 0.5)

    if frame_labels:
        ax.set_xticks(frames)
        ax.set_xticklabels(frame_labels, rotation=90, fontsize=6)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
