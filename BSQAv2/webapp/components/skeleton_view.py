"""Skeleton visualization helpers."""
from __future__ import annotations

from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .bootstrap import ensure_project_imports

ensure_project_imports()

from src.data.skeleton import BODY_SKELETON_EDGES, CRITICAL_KEYPOINTS


def skeleton_figure(
    keypoints: np.ndarray,
    frame_index: int,
    edges: Iterable[Tuple[int, int]] = BODY_SKELETON_EDGES,
):
    """Create a Matplotlib skeleton figure for one raw pixel frame."""
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.ndim != 3 or keypoints.shape[1:] != (17, 2):
        raise ValueError(f"keypoints must have shape (T, 17, 2); got {keypoints.shape}")

    frame_index = int(np.clip(frame_index, 0, keypoints.shape[0] - 1))
    frame = keypoints[frame_index]
    valid = ~(np.isclose(frame, 0.0).all(axis=1))

    fig, ax = plt.subplots(figsize=(5, 6))
    for start, end in edges:
        if valid[start] and valid[end]:
            ax.plot(
                [frame[start, 0], frame[end, 0]],
                [frame[start, 1], frame[end, 1]],
                linewidth=2,
                color="#4C78A8",
            )

    normal = [idx for idx in range(frame.shape[0]) if valid[idx] and idx not in CRITICAL_KEYPOINTS]
    critical = [idx for idx in range(frame.shape[0]) if valid[idx] and idx in CRITICAL_KEYPOINTS]

    if normal:
        ax.scatter(frame[normal, 0], frame[normal, 1], s=25, color="#72B7B2", label="joint")
    if critical:
        ax.scatter(frame[critical, 0], frame[critical, 1], s=45, color="#E45756", label="critical")

    ax.set_title(f"Skeleton frame {frame_index}")
    ax.set_xlabel("x pixel")
    ax.set_ylabel("y pixel")
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    if normal or critical:
        ax.legend(loc="best")
    fig.tight_layout()
    return fig
