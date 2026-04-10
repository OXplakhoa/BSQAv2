"""
MediaPipe Pose → COCO-17 CSV extraction pipeline.
Processes video clips and outputs skeleton CSV files matching v1 schema.
Includes quality control: confidence filtering, jump detection, missing joint checks.
Flagged clips are moved to data/flagged/ for manual review.
"""
import argparse
import csv
import shutil
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

try:
    import cv2
    import urllib.request
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ImportError:
    print("Install dependencies: pip install opencv-python mediapipe")
    raise


MODEL_PATH = Path("pose_landmarker_full.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"

def ensure_model():
    if not MODEL_PATH.exists():
        print("Downloading MediaPipe Pose model (heavy)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


# ── MediaPipe 33 → COCO-17 mapping ──────────────────────────────────────────
# MediaPipe has 33 landmarks, COCO uses 17 keypoints.
# This maps MediaPipe landmark indices to COCO-17 indices.
MEDIAPIPE_TO_COCO17 = {
    0: 0,   # nose → nose
    2: 1,   # left_eye_inner → left_eye (approx)
    5: 2,   # right_eye_inner → right_eye (approx)
    7: 3,   # left_ear → left_ear
    8: 4,   # right_ear → right_ear
    11: 5,  # left_shoulder → left_shoulder
    12: 6,  # right_shoulder → right_shoulder
    13: 7,  # left_elbow → left_elbow
    14: 8,  # right_elbow → right_elbow
    15: 9,  # left_wrist → left_wrist
    16: 10, # right_wrist → right_wrist
    23: 11, # left_hip → left_hip
    24: 12, # right_hip → right_hip
    25: 13, # left_knee → left_knee
    26: 14, # right_knee → right_knee
    27: 15, # left_ankle → left_ankle
    28: 16, # right_ankle → right_ankle
}

# Critical joints for badminton
# Note: Only shoulders are truly "critical" for QC. Wrists/elbows are
# frequently occluded or blurred in broadcast footage but still get
# reasonable estimates. We keep them for extraction but don't gate on them.
CRITICAL_COCO_INDICES = [5, 6]  # shoulders only — stable even in fast motion
NUM_COCO_KEYPOINTS = 17
CONFIDENCE_THRESHOLD = 0.3      # lowered: broadcast video has more blur
JUMP_THRESHOLD_PX = 200         # raised: jump smash can exceed 150px/frame
MISSING_JOINT_RATIO_LIMIT = 0.50


def extract_keypoints_from_video(
    video_path: str,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Run MediaPipe Pose on a video and extract COCO-17 keypoints per frame.

    Args:
        video_path: Path to video file
        min_detection_confidence: MediaPipe detection threshold
        min_tracking_confidence: MediaPipe tracking threshold

    Returns:
        keypoints: (T, 17, 2) array of (x_pixel, y_pixel)
        visibilities: (T, 17) array of visibility scores
        fps: Video FPS
    """
    ensure_model()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    all_keypoints = []
    all_visibilities = []

    base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=min_detection_confidence,
        min_pose_presence_confidence=min_tracking_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx * 1000 / fps)
            frame_idx += 1

            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            frame_kpts = np.zeros((NUM_COCO_KEYPOINTS, 2))
            frame_vis = np.zeros(NUM_COCO_KEYPOINTS)

            if results.pose_landmarks and len(results.pose_landmarks) > 0:
                pose_landmarks = results.pose_landmarks[0]
                for mp_idx, coco_idx in MEDIAPIPE_TO_COCO17.items():
                    lm = pose_landmarks[mp_idx]
                    frame_vis[coco_idx] = lm.visibility
                    if lm.visibility >= CONFIDENCE_THRESHOLD:
                        frame_kpts[coco_idx, 0] = lm.x * width
                        frame_kpts[coco_idx, 1] = lm.y * height

            all_keypoints.append(frame_kpts)
            all_visibilities.append(frame_vis)

    cap.release()

    keypoints = np.array(all_keypoints)   # (T, 17, 2)
    visibilities = np.array(all_visibilities)  # (T, 17)

    return keypoints, visibilities, fps


def quality_control(
    keypoints: np.ndarray,
    visibilities: np.ndarray,
    video_name: str = "",
) -> Tuple[bool, List[str]]:
    """
    Run quality control checks on extracted keypoints.

    Returns:
        (passed, reasons): True if passed, list of failure reasons
    """
    reasons = []
    T = keypoints.shape[0]

    if T < 5:
        reasons.append(f"Too few frames: {T}")
        return False, reasons

    # Check 1: Critical joint visibility across frames
    for coco_idx in CRITICAL_COCO_INDICES:
        visible_frames = np.sum(visibilities[:, coco_idx] >= CONFIDENCE_THRESHOLD)
        ratio = visible_frames / T
        if ratio < 0.5:
            reasons.append(
                f"Critical joint {coco_idx} visible only {ratio:.0%} of frames"
            )

    # Check 2: Relative jump detection — flag only when individual joints
    # deviate significantly from the body's median movement.
    # Real athletic movement (lunge, jump smash) moves ALL joints together;
    # tracker errors make 1-2 joints teleport while the rest stay coherent.
    #
    # EXCLUDED from check: limb extremities that naturally move much faster
    # than the torso during badminton strokes (wrists swing racket, ankles
    # lunge, knees/elbows follow through).
    OUTLIER_RATIO = 3.0   # joint must move >3x the median body displacement
    MIN_ABSOLUTE_PX = 150 # ignore small movements even if ratio is high
    JUMP_EXCLUDE_JOINTS = {7, 8, 9, 10, 13, 14, 15, 16}  # elbows, wrists, knees, ankles
    for t in range(1, T):
        diff = np.linalg.norm(keypoints[t] - keypoints[t-1], axis=1)
        valid = (keypoints[t].sum(axis=1) != 0) & (keypoints[t-1].sum(axis=1) != 0)
        if valid.sum() < 3:
            continue  # not enough joints to judge

        valid_diffs = diff[valid]
        median_disp = np.median(valid_diffs)

        for j_idx in np.where(valid)[0]:
            if j_idx in JUMP_EXCLUDE_JOINTS:
                continue  # skip limb extremities
            joint_disp = diff[j_idx]
            # Flag only if: (a) large absolute move AND (b) outlier vs body median
            if joint_disp > MIN_ABSOLUTE_PX and median_disp > 0 and (joint_disp / median_disp) > OUTLIER_RATIO:
                reasons.append(
                    f"Outlier joint {j_idx} at frame {t}: "
                    f"{joint_disp:.0f}px vs median {median_disp:.0f}px "
                    f"(ratio {joint_disp/median_disp:.1f}x)"
                )

    # Check 3: Overall missing joint ratio
    total_joints = T * NUM_COCO_KEYPOINTS
    missing = np.sum(keypoints.sum(axis=2) == 0)
    missing_ratio = missing / total_joints
    if missing_ratio > MISSING_JOINT_RATIO_LIMIT:
        reasons.append(f"Missing joint ratio: {missing_ratio:.0%} (limit: {MISSING_JOINT_RATIO_LIMIT:.0%})")

    passed = len(reasons) == 0
    return passed, reasons


def keypoints_to_csv_rows(
    keypoints: np.ndarray,
    sample_id: int,
    stroke_type: str,
) -> List[dict]:
    """
    Convert (T, 17, 2) keypoints to list of CSV row dicts matching v1 schema.
    Columns: id, type_of_shot, frame_count, kpt_0_x, kpt_0_y, ..., kpt_16_x, kpt_16_y
    """
    rows = []
    for t in range(keypoints.shape[0]):
        row = {
            "id": sample_id,
            "type_of_shot": stroke_type,
            "frame_count": t,
        }
        for j in range(NUM_COCO_KEYPOINTS):
            row[f"kpt_{j}_x"] = keypoints[t, j, 0]
            row[f"kpt_{j}_y"] = keypoints[t, j, 1]
        rows.append(row)
    return rows


def get_csv_header() -> List[str]:
    """Return CSV header matching v1 schema."""
    header = ["id", "type_of_shot", "frame_count"]
    for j in range(NUM_COCO_KEYPOINTS):
        header.extend([f"kpt_{j}_x", f"kpt_{j}_y"])
    return header


def process_clips_directory(
    clips_dir: str,
    output_dir: str,
    flagged_dir: str = "data/flagged",
    start_id: int = 1000,
    verify: bool = False,
) -> None:
    """
    Process all video clips in a directory structure:
      clips_dir/{stroke_type}/*.mp4 → output_dir/{stroke_type}_v2.csv

    Flagged clips (QC failures) are copied to flagged_dir with a log.

    Args:
        clips_dir: Root directory containing stroke_type subdirectories
        output_dir: Output directory for CSVs (e.g. data/youtube/)
        flagged_dir: Directory for QC-failed clips
        start_id: Starting sample ID for new data
        verify: If True, print summary stats per clip
    """
    clips_dir = Path(clips_dir)
    output_dir = Path(output_dir)
    flagged_dir = Path(flagged_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    flagged_dir.mkdir(parents=True, exist_ok=True)

    current_id = start_id
    total_passed = 0
    total_flagged = 0

    # Open a flagged log
    flag_log_path = flagged_dir / "qc_log.txt"
    flag_log = open(flag_log_path, 'a')

    for stroke_dir in sorted(clips_dir.iterdir()):
        if not stroke_dir.is_dir():
            continue

        stroke_type = stroke_dir.name.lower()
        video_files = sorted(
            [f for f in stroke_dir.iterdir() if f.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv')]
        )

        if not video_files:
            continue

        print(f"\n── Processing {stroke_type}: {len(video_files)} clips ──")

        csv_path = output_dir / f"{stroke_type}_v2.csv"
        all_rows = []

        # Load existing data if CSV already exists
        if csv_path.exists():
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                all_rows = list(reader)
            existing_ids = {int(r['id']) for r in all_rows}
            current_id = max(existing_ids) + 1 if existing_ids else current_id
            print(f"  Appending to existing CSV ({len(all_rows)} rows, next_id={current_id})")

        for vf in video_files:
            print(f"  → {vf.name}...", end=" ", flush=True)
            try:
                keypoints, visibilities, fps = extract_keypoints_from_video(str(vf))
            except Exception as e:
                print(f"ERROR: {e}")
                continue

            passed, reasons = quality_control(keypoints, visibilities, vf.name)

            if passed:
                rows = keypoints_to_csv_rows(keypoints, current_id, stroke_type)
                all_rows.extend(rows)
                print(f"✓ ({keypoints.shape[0]} frames, id={current_id})")
                current_id += 1
                total_passed += 1
            else:
                # Move to flagged directory
                flagged_stroke_dir = flagged_dir / stroke_type
                flagged_stroke_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(vf, flagged_stroke_dir / vf.name)

                reason_str = "; ".join(reasons)
                flag_log.write(f"{vf.name}: {reason_str}\n")
                print(f"⚠ FLAGGED ({reason_str})")
                total_flagged += 1

        # Write CSV
        if all_rows:
            header = get_csv_header()
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                writer.writerows(all_rows)
            print(f"  Saved: {csv_path} ({len(all_rows)} total rows)")

    flag_log.close()

    print(f"\n{'='*50}")
    print(f"  Passed QC: {total_passed}")
    print(f"  Flagged:   {total_flagged} (see {flag_log_path})")
    print(f"{'='*50}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract skeleton keypoints from video clips using MediaPipe"
    )
    parser.add_argument(
        "--input", type=str, default="data/clips",
        help="Input directory containing stroke_type subdirs with video clips"
    )
    parser.add_argument(
        "--output", type=str, default="data/youtube",
        help="Output directory for CSV files"
    )
    parser.add_argument(
        "--flagged", type=str, default="data/flagged",
        help="Directory for QC-failed clips (kept for manual review)"
    )
    parser.add_argument(
        "--start-id", type=int, default=1000,
        help="Starting sample ID for new YouTube data"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Print detailed stats per clip"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_clips_directory(
        clips_dir=args.input,
        output_dir=args.output,
        flagged_dir=args.flagged,
        start_id=args.start_id,
        verify=args.verify,
    )
