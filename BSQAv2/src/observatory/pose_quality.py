"""Pose quality diagnostics for Pipeline Observatory artifacts."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


JOINT_NAMES = {
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
}
CRITICAL_JOINTS = [5, 6, 7, 8, 9, 10]


def _missing_mask(
    keypoints: np.ndarray,
    visibilities: Optional[np.ndarray],
    visibility_threshold: float,
) -> np.ndarray:
    zero_coords = np.isclose(keypoints, 0.0).all(axis=2)
    if visibilities is None:
        return zero_coords
    low_visibility = np.asarray(visibilities) < visibility_threshold
    return zero_coords | low_visibility


def _label(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def evaluate_pose_quality(
    keypoints: np.ndarray,
    visibilities: Optional[np.ndarray] = None,
    visibility_threshold: float = 0.3,
    jump_threshold_px: float = 200.0,
) -> Dict:
    """Evaluate pose reliability for raw COCO-17 pixel keypoints.

    Args:
        keypoints: Array shaped (T, 17, 2).
        visibilities: Optional MediaPipe-derived visibility array (T, 17).
        visibility_threshold: Visibility below this is treated as unreliable.
        jump_threshold_px: Frame-to-frame displacement above this is an outlier.

    Returns:
        JSON-safe dictionary of metrics, score, label, and warning messages.
    """
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.ndim != 3 or keypoints.shape[1:] != (17, 2):
        raise ValueError(f"keypoints must have shape (T, 17, 2); got {keypoints.shape}")

    if visibilities is not None:
        visibilities = np.asarray(visibilities, dtype=np.float32)
        if visibilities.shape != keypoints.shape[:2]:
            raise ValueError(
                "visibilities must have shape (T, 17); "
                f"got {visibilities.shape}"
            )

    missing = _missing_mask(keypoints, visibilities, visibility_threshold)
    missing_joint_ratio = missing.mean(axis=1)
    missing_ratio = float(missing.mean())

    if visibilities is None:
        joint_visibility = 1.0 - missing.mean(axis=0)
    else:
        joint_visibility = np.where(missing, 0.0, visibilities).mean(axis=0)
    critical_visibility = {
        JOINT_NAMES[j]: float(joint_visibility[j]) for j in CRITICAL_JOINTS
    }
    critical_mean = float(np.mean([critical_visibility[JOINT_NAMES[j]] for j in CRITICAL_JOINTS]))

    diffs = np.linalg.norm(np.diff(keypoints, axis=0), axis=2)
    valid_pairs = ~(missing[1:] | missing[:-1])
    valid_diffs = diffs[valid_pairs]
    mean_jitter = float(np.mean(valid_diffs)) if valid_diffs.size else 0.0
    max_jump = float(np.max(valid_diffs)) if valid_diffs.size else 0.0
    outlier_jump_count = int(np.sum(valid_diffs > jump_threshold_px)) if valid_diffs.size else 0

    warnings: List[str] = []
    if missing_ratio > 0.2:
        warnings.append(f"High missing joint ratio: {missing_ratio:.1%}")
    for name, value in critical_visibility.items():
        if value < 0.55:
            warnings.append(f"Low critical joint visibility: {name} ({value:.2f})")
    if outlier_jump_count > 0:
        warnings.append(
            f"Detected {outlier_jump_count} outlier joint jump(s) above {jump_threshold_px:.0f}px"
        )

    missing_penalty = min(0.35, missing_ratio * 1.5)
    critical_penalty = min(0.30, max(0.0, 0.80 - critical_mean) * 0.9)
    low_critical_penalty = 0.20 if any(v < 0.55 for v in critical_visibility.values()) else 0.0
    jump_penalty = min(0.25, outlier_jump_count * 0.05)
    reliability_score = 1.0 - missing_penalty - critical_penalty - low_critical_penalty - jump_penalty
    reliability_score = float(np.clip(reliability_score, 0.0, 1.0))

    return {
        "reliability_score": reliability_score,
        "reliability_label": _label(reliability_score),
        "warnings": warnings,
        "missing_ratio": missing_ratio,
        "missing_joint_ratio_by_frame": missing_joint_ratio.astype(float).tolist(),
        "critical_joint_visibility": critical_visibility,
        "critical_joint_visibility_mean": critical_mean,
        "mean_jitter_px": mean_jitter,
        "max_jump_px": max_jump,
        "outlier_jump_count": outlier_jump_count,
        "visibility_threshold": float(visibility_threshold),
        "jump_threshold_px": float(jump_threshold_px),
    }
