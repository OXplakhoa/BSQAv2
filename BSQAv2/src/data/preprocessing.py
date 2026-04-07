"""
Preprocessing functions for skeleton data
- Normalize coordinates (hip-centered, torso-scaled)
- Handle missing keypoints (interpolation)
- Pad/truncate to fixed length
"""
import numpy as np
from ..config import (
    HIP_LEFT_IDX, HIP_RIGHT_IDX,
    SHOULDER_LEFT_IDX, SHOULDER_RIGHT_IDX,
    NUM_KEYPOINTS, COORD_DIM
)


def normalize_skeleton(keypoints: np.ndarray) -> np.ndarray:
    """
    Normalize skeleton coordinates:
    1. Center at hip midpoint
    2. Scale by torso length (hip to shoulder distance)

    Args:
        keypoints: (T, 17, 2) or (17, 2) array

    Returns:
        Normalized keypoints in roughly [-1, 1] range
    """
    original_shape = keypoints.shape
    if keypoints.ndim == 2:
        keypoints = keypoints[np.newaxis, ...]

    T, N, C = keypoints.shape
    normalized = np.zeros_like(keypoints)

    for t in range(T):
        frame = keypoints[t]

        hip_left = frame[HIP_LEFT_IDX]
        hip_right = frame[HIP_RIGHT_IDX]

        if np.allclose(hip_left, 0) and np.allclose(hip_right, 0):
            normalized[t] = frame
            continue

        hip_center = (hip_left + hip_right) / 2

        shoulder_left = frame[SHOULDER_LEFT_IDX]
        shoulder_right = frame[SHOULDER_RIGHT_IDX]
        shoulder_center = (shoulder_left + shoulder_right) / 2

        torso_length = np.linalg.norm(shoulder_center - hip_center)
        if torso_length < 1e-6:
            torso_length = 1.0

        centered = frame - hip_center
        normalized[t] = centered / torso_length

    if len(original_shape) == 2:
        return normalized[0]
    return normalized


def interpolate_missing(keypoints: np.ndarray, missing_val: float = 0.0) -> np.ndarray:
    """
    Interpolate missing keypoints (marked as 0.0) from adjacent frames.

    Args:
        keypoints: (T, 17, 2) array
        missing_val: Value indicating missing data

    Returns:
        Keypoints with missing values interpolated
    """
    T, N, C = keypoints.shape
    result = keypoints.copy()

    for joint_idx in range(N):
        for coord_idx in range(C):
            values = keypoints[:, joint_idx, coord_idx]
            missing_mask = np.isclose(values, missing_val)

            if missing_mask.all() or not missing_mask.any():
                continue

            valid_indices = np.where(~missing_mask)[0]
            missing_indices = np.where(missing_mask)[0]

            result[missing_indices, joint_idx, coord_idx] = np.interp(
                missing_indices,
                valid_indices,
                values[valid_indices]
            )

    return result


def pad_or_truncate(keypoints: np.ndarray, target_length: int) -> np.ndarray:
    """
    Pad (with zeros) or truncate sequence to target length.
    Uses center-cropping for truncation.

    Args:
        keypoints: (T, 17, 2) array
        target_length: Desired sequence length

    Returns:
        (target_length, 17, 2) array
    """
    T = keypoints.shape[0]

    if T == target_length:
        return keypoints
    elif T > target_length:
        start = (T - target_length) // 2
        return keypoints[start:start + target_length]
    else:
        pad_total = target_length - T
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        return np.pad(
            keypoints,
            ((pad_before, pad_after), (0, 0), (0, 0)),
            mode='constant',
            constant_values=0
        )


def preprocess_sequence(keypoints: np.ndarray, target_length: int = 64) -> np.ndarray:
    """
    Full preprocessing pipeline:
    1. Interpolate missing keypoints
    2. Normalize (hip-centered, torso-scaled)
    3. Pad/truncate to target length

    Args:
        keypoints: (T, 17, 2) raw keypoint coordinates
        target_length: Desired output sequence length

    Returns:
        (target_length, 17, 2) preprocessed keypoints
    """
    keypoints = interpolate_missing(keypoints)
    keypoints = normalize_skeleton(keypoints)
    keypoints = pad_or_truncate(keypoints, target_length)
    return keypoints
