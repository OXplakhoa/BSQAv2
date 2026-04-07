"""
Data augmentation techniques for skeleton sequences — v2
All functions: input (64, 17, 2) → output (64, 17, 2)

CRITICAL: time_warp and frame_dropout change intermediate frame count
but ALWAYS resample back to exactly 64 frames before returning.
This guarantees shape invariance for the model input.
"""
import numpy as np
from typing import List, Callable, Optional

from ..config import SEQUENCE_LENGTH, NUM_KEYPOINTS, COORD_DIM

# Left-right keypoint swap pairs (COCO-17 format)
# left_eye(1) ↔ right_eye(2), left_ear(3) ↔ right_ear(4),
# left_shoulder(5) ↔ right_shoulder(6), left_elbow(7) ↔ right_elbow(8),
# left_wrist(9) ↔ right_wrist(10), left_hip(11) ↔ right_hip(12),
# left_knee(13) ↔ right_knee(14), left_ankle(15) ↔ right_ankle(16)
LR_SWAP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


def _resample_frames(keypoints: np.ndarray, target_length: int) -> np.ndarray:
    """
    Resample a sequence to target_length frames using linear interpolation.

    Args:
        keypoints: (T, 17, 2) array
        target_length: desired output frame count

    Returns:
        (target_length, 17, 2) resampled array
    """
    T, N, C = keypoints.shape
    if T == target_length:
        return keypoints

    # Flatten to (T, 34) for interpolation, then reshape back
    flat = keypoints.reshape(T, -1)  # (T, 34)

    old_indices = np.linspace(0, 1, T)
    new_indices = np.linspace(0, 1, target_length)

    resampled = np.zeros((target_length, N * C))
    for col in range(flat.shape[1]):
        resampled[:, col] = np.interp(new_indices, old_indices, flat[:, col])

    return resampled.reshape(target_length, N, C)


def time_warp(
    keypoints: np.ndarray,
    speed_range: tuple = (0.8, 1.2),
    target_length: int = SEQUENCE_LENGTH,
) -> np.ndarray:
    """
    Time warping: change playback speed randomly, then resample to 64 frames.

    Speed < 1.0 = slow motion (more intermediate frames)
    Speed > 1.0 = fast forward (fewer intermediate frames)

    Args:
        keypoints: (64, 17, 2) input
        speed_range: (min_speed, max_speed)
        target_length: output frame count (always 64)

    Returns:
        (64, 17, 2) time-warped sequence
    """
    T = keypoints.shape[0]
    speed = np.random.uniform(*speed_range)

    # Warped length (before resampling back)
    warped_length = max(int(T / speed), 3)

    # Resample to warped length, then back to target
    warped = _resample_frames(keypoints, warped_length)
    return _resample_frames(warped, target_length)


def mirror_flip(keypoints: np.ndarray) -> np.ndarray:
    """
    Horizontal flip: negate x-coordinates and swap left/right keypoints.
    Simulates seeing the player from the opposite side.

    Args:
        keypoints: (64, 17, 2) input

    Returns:
        (64, 17, 2) mirrored sequence
    """
    flipped = keypoints.copy()

    # Negate x-coordinates (column 0)
    flipped[:, :, 0] = -flipped[:, :, 0]

    # Swap left/right keypoint pairs
    for left_idx, right_idx in LR_SWAP_PAIRS:
        temp = flipped[:, left_idx, :].copy()
        flipped[:, left_idx, :] = flipped[:, right_idx, :]
        flipped[:, right_idx, :] = temp

    return flipped


def joint_noise(
    keypoints: np.ndarray,
    sigma: float = 0.015,
) -> np.ndarray:
    """
    Add small Gaussian noise to joint coordinates.
    Simulates slight pose estimation inaccuracies.

    Only applies noise to non-zero joints (preserves missing markers).

    Args:
        keypoints: (64, 17, 2) input
        sigma: noise standard deviation (relative to normalized coordinates)

    Returns:
        (64, 17, 2) noisy sequence
    """
    noisy = keypoints.copy()
    noise = np.random.normal(0, sigma, keypoints.shape).astype(np.float32)

    # Only add noise to non-zero joints
    non_zero = keypoints.sum(axis=2) != 0  # (T, 17)
    non_zero_3d = np.expand_dims(non_zero, axis=2)  # (T, 17, 1)
    noisy += noise * non_zero_3d

    return noisy


def frame_dropout(
    keypoints: np.ndarray,
    dropout_rate: float = 0.08,
    target_length: int = SEQUENCE_LENGTH,
) -> np.ndarray:
    """
    Randomly drop frames, then interpolate + resample back to 64 frames.
    Simulates video frame loss or variable frame rates.

    Args:
        keypoints: (64, 17, 2) input
        dropout_rate: fraction of frames to drop (0.05-0.10)
        target_length: output frame count (always 64)

    Returns:
        (64, 17, 2) sequence with frames dropped and resampled
    """
    T = keypoints.shape[0]
    n_drop = max(1, int(T * dropout_rate))

    # Random frame indices to keep
    drop_indices = set(np.random.choice(range(1, T - 1), size=n_drop, replace=False))
    kept = np.array([keypoints[i] for i in range(T) if i not in drop_indices])

    # Resample back to target length
    return _resample_frames(kept, target_length)


def compose_augmentations(
    augmentation_fns: List[Callable],
) -> Callable:
    """
    Compose multiple augmentation functions into a single callable.

    Usage:
        aug_fn = compose_augmentations([time_warp, mirror_flip, joint_noise])
        augmented = aug_fn(keypoints)

    Args:
        augmentation_fns: List of augmentation functions

    Returns:
        Composed function that applies all augmentations sequentially
    """
    def composed(keypoints: np.ndarray) -> np.ndarray:
        result = keypoints.copy()
        for fn in augmentation_fns:
            result = fn(result)
        return result
    return composed


def random_augmentation(
    keypoints: np.ndarray,
    p_time_warp: float = 0.5,
    p_mirror: float = 0.5,
    p_noise: float = 0.7,
    p_dropout: float = 0.3,
) -> np.ndarray:
    """
    Apply random augmentations with given probabilities.
    Convenient for on-the-fly augmentation during training.

    Args:
        keypoints: (64, 17, 2) input
        p_*: probability of applying each augmentation

    Returns:
        (64, 17, 2) augmented sequence
    """
    result = keypoints.copy()

    if np.random.random() < p_time_warp:
        result = time_warp(result)
    if np.random.random() < p_mirror:
        result = mirror_flip(result)
    if np.random.random() < p_noise:
        result = joint_noise(result)
    if np.random.random() < p_dropout:
        result = frame_dropout(result)

    return result


if __name__ == "__main__":
    # Quick shape verification test
    print("Testing augmentation functions...")
    x = np.random.randn(SEQUENCE_LENGTH, NUM_KEYPOINTS, COORD_DIM).astype(np.float32)

    for fn_name, fn in [
        ("time_warp", time_warp),
        ("mirror_flip", mirror_flip),
        ("joint_noise", joint_noise),
        ("frame_dropout", frame_dropout),
        ("random_augmentation", random_augmentation),
    ]:
        out = fn(x.copy())
        assert out.shape == (SEQUENCE_LENGTH, NUM_KEYPOINTS, COORD_DIM), \
            f"{fn_name}: expected {(SEQUENCE_LENGTH, NUM_KEYPOINTS, COORD_DIM)}, got {out.shape}"
        print(f"  ✓ {fn_name}: {x.shape} → {out.shape}")

    # Test compose
    composed = compose_augmentations([time_warp, mirror_flip, joint_noise])
    out = composed(x.copy())
    assert out.shape == (SEQUENCE_LENGTH, NUM_KEYPOINTS, COORD_DIM)
    print(f"  ✓ compose: {x.shape} → {out.shape}")

    print("\n✓ All augmentation tests passed!")
