"""
Preprocessing functions for skeleton data
- Normalize coordinates (hip-centered, torso-scaled)
- Handle missing keypoints (interpolation)
- Pad/truncate to fixed length
"""
import numpy as np
import torch
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
    Pad or truncate sequence to target length.
    Uses center-cropping for truncation, edge-replication for padding
    (avoids zero frames that confuse the model).

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
        # Repeat first and last frames instead of zeros
        pad_total = target_length - T
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        # Use first frame for front padding, last frame for back padding
        front_pad = np.repeat(keypoints[:1], pad_before, axis=0) if pad_before > 0 else np.zeros((0, *keypoints.shape[1:]))
        back_pad = np.repeat(keypoints[-1:], pad_after, axis=0) if pad_after > 0 else np.zeros((0, *keypoints.shape[1:]))
        return np.concatenate([front_pad, keypoints, back_pad], axis=0)


def preprocess_sequence(keypoints: np.ndarray, target_length: int = 64) -> np.ndarray:
    """
    Full preprocessing pipeline:
    0. Replace NaN/inf with 0.0 (missing marker for interpolation)
    1. Interpolate missing keypoints
    2. Normalize (hip-centered, torso-scaled)
    3. Pad/truncate to target length

    Args:
        keypoints: (T, 17, 2) raw keypoint coordinates
        target_length: Desired output sequence length

    Returns:
        (target_length, 17, 2) preprocessed keypoints
    """
    keypoints = np.nan_to_num(keypoints, nan=0.0, posinf=0.0, neginf=0.0)
    keypoints = interpolate_missing(keypoints)
    keypoints = normalize_skeleton(keypoints)
    keypoints = pad_or_truncate(keypoints, target_length)
    return keypoints


def compute_velocity(keypoints: np.ndarray) -> np.ndarray:
    """
    Add per-joint velocity channels via frame-to-frame finite difference.

    Args:
        keypoints: (T, 17, 2) position coordinates

    Returns:
        (T, 17, 4) concatenated [position, velocity]
    """
    T, N, C = keypoints.shape
    vel = np.zeros_like(keypoints)
    vel[1:] = keypoints[1:] - keypoints[:-1]
    return np.concatenate([keypoints, vel], axis=-1)


def add_velocity_torch(x: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    """
    Append per-joint velocity to input tensor (GPU-compatible).

    Args:
        x: (B, T, N, 2) position coordinates
        normalize: DISABLED by default — per-batch normalization destroys
                   the speed magnitude signal critical for stroke classification
                   (smash=fast, net_shot=slow).

    Returns:
        (B, T, N, 4) concatenated [position, velocity]
    """
    vel = x[:, 1:] - x[:, :-1]
    vel = torch.nn.functional.pad(vel, (0, 0, 0, 0, 1, 0, 0, 0))
    if normalize:
        vel_std = vel.std(dim=(1, 2, 3), keepdim=True) + 1e-8
        pos_std = x.std(dim=(1, 2, 3), keepdim=True) + 1e-8
        vel = vel * (pos_std / vel_std)
    return torch.cat([x, vel], dim=-1)
