"""
Visualization utilities for skeleton and training
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
    
    # Plot edges
    for start, end in SKELETON_EDGES:
        x = [keypoints[start, 0], keypoints[end, 0]]
        y = [keypoints[start, 1], keypoints[end, 1]]
        # Skip if either point is missing (0, 0)
        if np.allclose(keypoints[start], 0) or np.allclose(keypoints[end], 0):
            continue
        ax.plot(x, y, 'b-', linewidth=2, alpha=0.7)
    
    # Plot joints
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
    ax.invert_yaxis()  # Image coordinates
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
    
    Args:
        train_losses: Training loss per epoch
        val_losses: Validation loss per epoch
        train_accs: Training accuracy per epoch
        val_accs: Validation accuracy per epoch
        save_path: Path to save figure (optional)
    
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss
    ax1.plot(epochs, train_losses, 'b-', label='Train')
    ax1.plot(epochs, val_losses, 'r-', label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(epochs, train_accs, 'b-', label='Train')
    ax2.plot(epochs, val_accs, 'r-', label='Val')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.85, color='g', linestyle='--', label='Target (85%)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
