"""Robustness experiment helpers for perturbing skeleton sequences."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np


WRIST_JOINTS = [9, 10]
ELBOW_JOINTS = [7, 8]


def degrade_keypoints(
    keypoints: np.ndarray,
    noise_std: float = 0.0,
    drop_wrists: bool = False,
    drop_elbows: bool = False,
    frame_dropout_rate: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """Apply deterministic artificial pose degradation to a skeleton sequence.

    Args:
        keypoints: ``(T, 17, 2)`` skeleton array.
        noise_std: Gaussian coordinate noise in the same coordinate space as input.
        drop_wrists/drop_elbows: zero selected critical joints for all frames.
        frame_dropout_rate: fraction of frames to zero out entirely.
        seed: deterministic RNG seed for presentation reproducibility.
    """
    arr = np.asarray(keypoints, dtype=np.float32).copy()
    if arr.ndim != 3 or arr.shape[1:] != (17, 2):
        raise ValueError(f"keypoints must have shape (T, 17, 2); got {arr.shape}")

    rng = np.random.default_rng(seed)
    if noise_std > 0:
        valid = ~(np.isclose(arr, 0.0).all(axis=2))
        noise = rng.normal(0.0, float(noise_std), size=arr.shape).astype(np.float32)
        arr[valid] += noise[valid]

    if drop_wrists:
        arr[:, WRIST_JOINTS, :] = 0.0
    if drop_elbows:
        arr[:, ELBOW_JOINTS, :] = 0.0

    frame_dropout_rate = float(np.clip(frame_dropout_rate, 0.0, 1.0))
    if frame_dropout_rate > 0 and arr.shape[0] > 0:
        n_drop = int(round(frame_dropout_rate * arr.shape[0]))
        n_drop = min(arr.shape[0], max(0, n_drop))
        if n_drop:
            indices = rng.choice(arr.shape[0], size=n_drop, replace=False)
            arr[indices, :, :] = 0.0

    return arr


def degradation_summary_rows(original: np.ndarray, degraded: np.ndarray) -> List[Dict[str, float]]:
    original = np.asarray(original, dtype=np.float32)
    degraded = np.asarray(degraded, dtype=np.float32)
    if original.shape != degraded.shape:
        raise ValueError(f"original and degraded shapes differ: {original.shape} vs {degraded.shape}")

    missing = np.isclose(degraded, 0.0).all(axis=2)
    original_valid = ~np.isclose(original, 0.0).all(axis=2)
    degraded_valid = ~missing
    both_valid = original_valid & degraded_valid

    if both_valid.any():
        shifts = np.linalg.norm(degraded[both_valid] - original[both_valid], axis=1)
        mean_shift = float(np.mean(shifts))
        max_shift = float(np.max(shifts))
    else:
        mean_shift = 0.0
        max_shift = 0.0

    frame_missing = missing.all(axis=1)
    return [
        {"metric": "missing_joint_ratio", "value": float(missing.mean())},
        {"metric": "dropped_frame_ratio", "value": float(frame_missing.mean())},
        {"metric": "mean_coordinate_shift", "value": mean_shift},
        {"metric": "max_coordinate_shift", "value": max_shift},
    ]


def prediction_delta_rows(
    baseline_probabilities: Dict[str, float],
    degraded_probabilities: Dict[str, float],
) -> List[Dict[str, float]]:
    labels = sorted(set(baseline_probabilities) | set(degraded_probabilities))
    rows: List[Dict[str, float]] = []
    for label in labels:
        baseline = float(baseline_probabilities.get(label, 0.0))
        degraded = float(degraded_probabilities.get(label, 0.0))
        delta = degraded - baseline
        rows.append({
            "class": label,
            "baseline_probability": baseline,
            "degraded_probability": degraded,
            "delta": delta,
            "absolute_delta": round(abs(delta), 12),
        })
    rows.sort(key=lambda row: (row["absolute_delta"], row["baseline_probability"]), reverse=True)
    return rows


def robustness_curve_rows(results: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = [
        {
            "severity": float(item.get("severity", 0.0)),
            "prediction": item.get("prediction", "missing"),
            "confidence": float(item.get("confidence", 0.0) or 0.0),
        }
        for item in results
    ]
    rows.sort(key=lambda row: row["severity"])
    return rows


def robustness_curve_figure(rows: Iterable[Dict[str, Any]]):
    """Create a Matplotlib confidence-vs-severity curve to avoid VegaLite warnings."""
    curve = robustness_curve_rows(rows)
    severities = [row["severity"] for row in curve]
    confidences = [row["confidence"] for row in curve]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(severities, confidences, marker="o", linewidth=2, color="#4C78A8")
    for row in curve:
        ax.annotate(
            str(row["prediction"]),
            (row["severity"], row["confidence"]),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
        )
    ax.set_title("RF confidence under coordinate noise")
    ax.set_xlabel("noise severity σ")
    ax.set_ylabel("RF confidence")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig
