"""
Biomechanical feature extraction from skeleton sequences.

Extracts joint angles, velocities, and spatial features from COCO-17
keypoint sequences. Output: one feature vector per clip for use in
Random Forest, entropy analysis, t-SNE clustering, etc.

COCO-17 joint indices:
    0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear,
    5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow,
    9=left_wrist, 10=right_wrist, 11=left_hip, 12=right_hip,
    13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── Joint index constants ─────────────────────────────────────────────────────
LS, RS = 5, 6      # shoulders
LE, RE = 7, 8      # elbows
LW, RW = 9, 10     # wrists
LH, RH = 11, 12    # hips
LK, RK = 13, 14    # knees
LA, RA = 15, 16    # ankles

# Joint triples for angle computation: (proximal, vertex, distal)
ANGLE_TRIPLES = {
    "right_elbow":   (RS, RE, RW),
    "left_elbow":    (LS, LE, LW),
    "right_knee":    (RH, RK, RA),
    "left_knee":     (LH, LK, LA),
    "right_shoulder":(RE, RS, RH),
    "left_shoulder": (LE, LS, LH),
    "right_hip":     (RS, RH, RK),
    "left_hip":      (LS, LH, LK),
}

# Keypoint pairs for velocity
VELOCITY_JOINTS = {
    "right_wrist": RW,
    "left_wrist":  LW,
    "right_elbow": RE,
    "left_elbow":  LE,
    "right_knee":  RK,
    "left_knee":   LK,
    "hip_center":  None,  # computed from LH+RH midpoint
}


# ── Core computation functions ────────────────────────────────────────────────

def _angle_between(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Angle in degrees between two vectors (last axis = xy)."""
    dot = np.sum(v1 * v2, axis=-1)
    norm = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
    norm = np.clip(norm, 1e-8, None)
    cos = np.clip(dot / norm, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def compute_joint_angle(
    keypoints: np.ndarray,
    proximal: int,
    vertex: int,
    distal: int,
) -> np.ndarray:
    """
    Compute angle at vertex joint over time.

    Args:
        keypoints: (T, 17, 2) array
        proximal, vertex, distal: joint indices

    Returns:
        (T,) array of angles in degrees. NaN where any joint is missing.
    """
    v_prox = keypoints[:, proximal] - keypoints[:, vertex]   # → vertex
    v_dist = keypoints[:, distal] - keypoints[:, vertex]      # → vertex
    angles = _angle_between(v_prox, v_dist)

    # Mask frames where any of the three joints is zero (missing)
    missing = (
        (np.sum(keypoints[:, proximal], axis=1) == 0) |
        (np.sum(keypoints[:, vertex], axis=1) == 0) |
        (np.sum(keypoints[:, distal], axis=1) == 0)
    )
    angles[missing] = np.nan
    return angles


def compute_torso_angle(keypoints: np.ndarray) -> np.ndarray:
    """
    Torso angle from vertical (0 = upright, positive = leaning right).

    Uses shoulder midpoint → hip midpoint vector.
    """
    shoulder_mid = (keypoints[:, LS] + keypoints[:, RS]) / 2
    hip_mid = (keypoints[:, LH] + keypoints[:, RH]) / 2
    torso_vec = shoulder_mid - hip_mid
    vertical = np.array([0.0, -1.0])
    vertical = np.tile(vertical, (torso_vec.shape[0], 1))
    angles = _angle_between(torso_vec, vertical)

    missing = (
        (np.sum(keypoints[:, LS], axis=1) == 0) |
        (np.sum(keypoints[:, RS], axis=1) == 0) |
        (np.sum(keypoints[:, LH], axis=1) == 0) |
        (np.sum(keypoints[:, RH], axis=1) == 0)
    )
    angles[missing] = np.nan
    return angles


def compute_joint_speed(keypoints: np.ndarray, joint_idx: int) -> np.ndarray:
    """
    Frame-to-frame Euclidean displacement for a single joint.

    Returns:
        (T,) array of speeds (magnitude). First frame = 0 prepended.
        NaN where joint is missing.
    """
    coords = keypoints[:, joint_idx, :]  # (T, 2)
    diff = np.linalg.norm(np.diff(coords, axis=0), axis=1)  # (T-1,)
    speed = np.concatenate([[0.0], diff])  # (T,) aligned with frames
    missing = np.sum(coords, axis=1) == 0
    speed[missing] = np.nan
    return speed


def compute_joint_velocity_components(
    keypoints: np.ndarray, joint_idx: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Frame-to-frame velocity components (dx, dy) for a single joint.

    Returns:
        (dx, dy): each (T,) arrays, first frame=0. NaN where joint missing.
    """
    coords = keypoints[:, joint_idx, :]  # (T, 2)
    d = np.diff(coords, axis=0)  # (T-1, 2)
    dx = np.concatenate([[0.0], d[:, 0]])
    dy = np.concatenate([[0.0], d[:, 1]])
    missing = np.sum(coords, axis=1) == 0
    dx[missing] = np.nan
    dy[missing] = np.nan
    return dx, dy


def compute_hip_center_speed(keypoints: np.ndarray) -> np.ndarray:
    """Speed of hip midpoint (body center proxy)."""
    hip_mid = (keypoints[:, LH] + keypoints[:, RH]) / 2
    diff = np.linalg.norm(np.diff(hip_mid, axis=0), axis=1)
    speed = np.concatenate([[0.0], diff])
    missing = (np.sum(keypoints[:, LH], axis=1) == 0) | (np.sum(keypoints[:, RH], axis=1) == 0)
    speed[missing] = np.nan
    return speed


def compute_torso_length(keypoints: np.ndarray) -> np.ndarray:
    """Hip-to-shoulder distance per frame."""
    shoulder_mid = (keypoints[:, LS] + keypoints[:, RS]) / 2
    hip_mid = (keypoints[:, LH] + keypoints[:, RH]) / 2
    length = np.linalg.norm(shoulder_mid - hip_mid, axis=1)
    missing = (
        (np.sum(keypoints[:, LS], axis=1) == 0) |
        (np.sum(keypoints[:, RS], axis=1) == 0) |
        (np.sum(keypoints[:, LH], axis=1) == 0) |
        (np.sum(keypoints[:, RH], axis=1) == 0)
    )
    length[missing] = np.nan
    return length


def compute_contact_height(keypoints: np.ndarray) -> float:
    """
    Estimate contact height: max wrist y minus hip y.
    Higher = arm raised more (smash/clear vs net shot).
    Returns NaN if wrist or hip is always missing.
    """
    hip_y = (keypoints[:, LH, 1] + keypoints[:, RH, 1]) / 2
    wrist_y = (keypoints[:, RW, 1] + keypoints[:, LW, 1]) / 2

    valid_hip = ~((np.sum(keypoints[:, LH], axis=1) == 0) | (np.sum(keypoints[:, RH], axis=1) == 0))
    valid_wrist = ~((np.sum(keypoints[:, RW], axis=1) == 0) | (np.sum(keypoints[:, LW], axis=1) == 0))
    valid = valid_hip & valid_wrist
    if not valid.any():
        return np.nan

    # Contact = frame where wrist is highest (lowest y = highest on screen)
    contact_frame = np.nanargmin(wrist_y[valid])
    contact_idx = np.where(valid)[0][contact_frame]
    return float(hip_y[contact_idx] - wrist_y[contact_idx])  # positive = above hip


def compute_knee_bend(keypoints: np.ndarray) -> float:
    """Maximum knee bend: min hip-knee Y distance. Larger = deeper bend."""
    right_bend = keypoints[:, RH, 1] - keypoints[:, RK, 1]
    left_bend = keypoints[:, LH, 1] - keypoints[:, LK, 1]
    valid_r = ~((np.sum(keypoints[:, RH], axis=1) == 0) | (np.sum(keypoints[:, RK], axis=1) == 0))
    valid_l = ~((np.sum(keypoints[:, LH], axis=1) == 0) | (np.sum(keypoints[:, LK], axis=1) == 0))
    vals = []
    if valid_r.any():
        vals.append(np.max(right_bend[valid_r]))
    if valid_l.any():
        vals.append(np.max(left_bend[valid_l]))
    return float(np.max(vals)) if vals else np.nan


# ── Statistical aggregation ───────────────────────────────────────────────────

def _safe_stats(arr: np.ndarray, prefix: str) -> Dict[str, float]:
    """Compute mean, max, min, std over time, ignoring NaN."""
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return {f"{prefix}_mean": np.nan, f"{prefix}_max": np.nan,
                f"{prefix}_min": np.nan, f"{prefix}_std": np.nan}
    return {
        f"{prefix}_mean": float(np.mean(valid)),
        f"{prefix}_max":  float(np.max(valid)),
        f"{prefix}_min":  float(np.min(valid)),
        f"{prefix}_std":  float(np.std(valid)),
    }


def _argmax_valid(arr: np.ndarray) -> float:
    """Index of maximum value (as fraction of sequence) ignoring NaN."""
    valid = ~np.isnan(arr)
    if not valid.any():
        return np.nan
    return float(np.argmax(np.where(valid, arr, -np.inf))) / len(arr)


def _range_motion(arr: np.ndarray) -> float:
    """Range of motion: max - min over valid frames."""
    valid = arr[~np.isnan(arr)]
    return float(np.max(valid) - np.min(valid)) if len(valid) > 0 else np.nan


# ── Main feature extraction ───────────────────────────────────────────────────

def extract_features(keypoints: np.ndarray) -> Dict[str, float]:
    """
    Extract biomechanical features from a single clip.

    Args:
        keypoints: (T, 17, 2) array of COCO-17 coordinates (pixel space, un-normalized OK)

    Returns:
        Dict of feature_name → scalar value
    """
    T = keypoints.shape[0]
    features: Dict[str, float] = {}

    # 1. Joint angles — per-frame → aggregate stats
    for name, (prox, vert, dist) in ANGLE_TRIPLES.items():
        angles = compute_joint_angle(keypoints, prox, vert, dist)
        features.update(_safe_stats(angles, f"angle_{name}"))
        # Range of motion
        features[f"angle_{name}_rom"] = _range_motion(angles)
        # Frame of max angle (extension)
        features[f"angle_{name}_maxframe"] = _argmax_valid(angles)

    # Torso angle
    torso_angles = compute_torso_angle(keypoints)
    features.update(_safe_stats(torso_angles, "torso_angle"))
    features["torso_angle_rom"] = _range_motion(torso_angles)

    # 2. Joint velocities (speed magnitude + directional components)
    DIR_JOINTS = {"right_wrist": RW, "left_wrist": LW,
                   "right_elbow": RE, "left_elbow": LE, "hip_center": None}
    for name, jidx in DIR_JOINTS.items():
        if jidx is None:
            speed = compute_hip_center_speed(keypoints)
        else:
            speed = compute_joint_speed(keypoints, jidx)
            dx, dy = compute_joint_velocity_components(keypoints, jidx)
            features.update(_safe_stats(dx, f"velx_{name}"))
            features.update(_safe_stats(dy, f"vely_{name}"))
            # Net displacement (start -> end position change)
            coords = keypoints[:, jidx, :]
            valid = ~(np.sum(coords, axis=1) == 0)
            if valid.any():
                vc = coords[valid]
                features[f"velx_{name}_netdisp"] = float(vc[-1, 0] - vc[0, 0])
                features[f"vely_{name}_netdisp"] = float(vc[-1, 1] - vc[0, 1])
            else:
                features[f"velx_{name}_netdisp"] = np.nan
                features[f"vely_{name}_netdisp"] = np.nan
        features.update(_safe_stats(speed, f"speed_{name}"))
        features[f"speed_{name}_maxframe"] = _argmax_valid(speed)

    # Racket-head speed proxy (right wrist speed)
    rw_speed = compute_joint_speed(keypoints, RW)
    lw_speed = compute_joint_speed(keypoints, LW)
    features.update(_safe_stats(rw_speed, "speed_racket_proxy"))
    features[f"speed_racket_proxy_maxframe"] = _argmax_valid(rw_speed)

    # Handedness: which wrist moves faster?
    rw_max = np.nanmax(rw_speed) if np.any(~np.isnan(rw_speed)) else 0
    lw_max = np.nanmax(lw_speed) if np.any(~np.isnan(lw_speed)) else 0
    features["hand_dominance"] = 1.0 if rw_max > lw_max else -1.0

    # 3. Spatial features
    features["contact_height"] = compute_contact_height(keypoints)
    features["knee_bend_max"] = compute_knee_bend(keypoints)

    # Torso length stats
    torso_len = compute_torso_length(keypoints)
    features.update(_safe_stats(torso_len, "torso_length"))

    # Hip lateral displacement (total horizontal movement)
    hip_x = (keypoints[:, LH, 0] + keypoints[:, RH, 0]) / 2
    valid_hip = ~((np.sum(keypoints[:, LH], axis=1) == 0) | (np.sum(keypoints[:, RH], axis=1) == 0))
    if valid_hip.any():
        hip_x_valid = hip_x[valid_hip]
        features["hip_x_displacement"] = float(np.max(hip_x_valid) - np.min(hip_x_valid))
    else:
        features["hip_x_displacement"] = np.nan

    # Vertical displacement (jump height proxy)
    hip_y = (keypoints[:, LH, 1] + keypoints[:, RH, 1]) / 2
    if valid_hip.any():
        hip_y_valid = hip_y[valid_hip]
        features["hip_y_displacement"] = float(np.max(hip_y_valid) - np.min(hip_y_valid))
    else:
        features["hip_y_displacement"] = np.nan

    # 4. Temporal features
    features["num_frames"] = T

    # Impact frame estimate: frame of max wrist speed
    features["impact_frame"] = _argmax_valid(rw_speed)

    # Preparation-to-impact ratio: where in the sequence does max wrist speed occur?
    # Late = smash/clear (wind-up then accelerate), early = net shot
    features["swing_phase_ratio"] = features["impact_frame"]

    # 5. Symmetry features
    for pair_name, (left_j, right_j) in [("wrist", (LW, RW)), ("elbow", (LE, RE)),
                                          ("knee", (LK, RK)), ("ankle", (LA, RA))]:
        left_y = keypoints[:, left_j, 1]
        right_y = keypoints[:, right_j, 1]
        valid = ~((np.sum(keypoints[:, left_j], axis=1) == 0) | (np.sum(keypoints[:, right_j], axis=1) == 0))
        if valid.any():
            diff = np.abs(left_y[valid] - right_y[valid])
            features[f"symmetry_{pair_name}_y"] = float(np.mean(diff))
        else:
            features[f"symmetry_{pair_name}_y"] = np.nan

    return features


# ── DataFrame builder ─────────────────────────────────────────────────────────

def build_feature_df(
    clips: List[Tuple[np.ndarray, int, str]],
) -> pd.DataFrame:
    """
    Build feature DataFrame from list of (keypoints, clip_id, stroke_type) tuples.

    Args:
        clips: List of (keypoints_array, id, stroke_label)

    Returns:
        DataFrame with one row per clip, columns = feature names + 'id' + 'stroke'
    """
    rows = []
    for kpts, cid, stroke in clips:
        feats = extract_features(kpts)
        feats["id"] = cid
        feats["stroke"] = stroke
        rows.append(feats)

    df = pd.DataFrame(rows)
    # Move id and stroke to front
    cols = ["id", "stroke"] + [c for c in df.columns if c not in ("id", "stroke")]
    return df[cols]


def build_feature_df_from_dataset(dataset) -> pd.DataFrame:
    """
    Build feature DataFrame from a BadmintonDataset.

    Keypoints are expected as (T, 17, 2) numpy arrays.
    """
    from src.config import IDX_TO_CLASS

    clips = []
    for i in range(len(dataset)):
        kpts, label = dataset[i]
        if hasattr(kpts, "numpy"):
            kpts = kpts.numpy()
        if isinstance(label, (int, np.integer)):
            stroke = IDX_TO_CLASS.get(int(label), str(label))
        else:
            stroke = str(label)
        clips.append((kpts, i, stroke))
    return build_feature_df(clips)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Extract biomechanical features from skeleton CSVs")
    parser.add_argument("--input", type=str, nargs="+", required=True,
                        help="CSV files or directories to process")
    parser.add_argument("--output", type=str, default="data/biomechanics_features.csv",
                        help="Output CSV path")
    return parser.parse_args()


if __name__ == "__main__":
    import sys
    _project_root = Path(__file__).parent.parent.parent  # src/data/ -> src/ -> BSQAv2/
    sys.path.insert(0, str(_project_root))

    from src.config import SEQUENCE_LENGTH
    from src.data.dataset import BadmintonDataset

    args = parse_args()

    data_dirs = [Path(d) for d in args.input]

    # Load all data through the existing dataset pipeline
    ds = BadmintonDataset(
        data_dirs=data_dirs,
        sequence_length=SEQUENCE_LENGTH,
        augment_fn=None,
    )
    print(f"Loaded {len(ds)} clips from {len(args.input)} source(s)")

    df = build_feature_df_from_dataset(ds)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Features saved: {args.output}")
    print(f"Shape: {df.shape[0]} clips × {df.shape[1]-2} features")
    print(f"Stroke distribution:\n{df['stroke'].value_counts().to_string()}")
