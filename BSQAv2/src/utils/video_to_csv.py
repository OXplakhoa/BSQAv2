"""
MediaPipe Pose -> COCO-17 CSV extraction pipeline.
Processes video clips and outputs skeleton CSV files matching v1 schema.
Includes quality control: confidence filtering, jump detection, missing joint checks.
Flagged clips are moved to data/flagged/ for manual review.
"""
import argparse
import csv
import json
import shutil
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

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


# -- MediaPipe 33 -> COCO-17 mapping ------------------------------------------
# MediaPipe has 33 landmarks, COCO uses 17 keypoints.
# This maps MediaPipe landmark indices to COCO-17 indices.
MEDIAPIPE_TO_COCO17 = {
    0: 0,   # nose -> nose
    2: 1,   # left_eye_inner -> left_eye (approx)
    5: 2,   # right_eye_inner -> right_eye (approx)
    7: 3,   # left_ear -> left_ear
    8: 4,   # right_ear -> right_ear
    11: 5,  # left_shoulder -> left_shoulder
    12: 6,  # right_shoulder -> right_shoulder
    13: 7,  # left_elbow -> left_elbow
    14: 8,  # right_elbow -> right_elbow
    15: 9,  # left_wrist -> left_wrist
    16: 10, # right_wrist -> right_wrist
    23: 11, # left_hip -> left_hip
    24: 12, # right_hip -> right_hip
    25: 13, # left_knee -> left_knee
    26: 14, # right_knee -> right_knee
    27: 15, # left_ankle -> left_ankle
    28: 16, # right_ankle -> right_ankle
}

# Critical joints for badminton
# Note: Only shoulders are truly "critical" for QC. Wrists/elbows are
# frequently occluded or blurred in broadcast footage but still get
# reasonable estimates. We keep them for extraction but don't gate on them.
CRITICAL_COCO_INDICES = [5, 6]  # shoulders only ? stable even in fast motion
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
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Run quality control checks on extracted keypoints.

    Returns:
        (passed, reasons, stats): passed flag, failure reasons, structured QC stats dict
    """
    reasons = []
    T = keypoints.shape[0]

    # -- Compute QC stats (always, regardless of pass/fail) --
    stats: Dict[str, Any] = {
        "video_name": video_name,
        "frame_count": T,
        "passed": True,  # will update
        "reasons": [],
    }

    # Per-joint mean visibility
    per_joint_mean_vis = {}
    per_joint_visible_ratio = {}
    for j in range(NUM_COCO_KEYPOINTS):
        vis_j = visibilities[:, j]
        per_joint_mean_vis[f"kpt_{j}_mean_vis"] = round(float(np.mean(vis_j)), 4)
        per_joint_visible_ratio[f"kpt_{j}_vis_ratio"] = round(
            float(np.sum(vis_j >= CONFIDENCE_THRESHOLD) / T), 4
        )
    stats.update(per_joint_mean_vis)
    stats.update(per_joint_visible_ratio)

    # Overall missing joint ratio
    total_joints = T * NUM_COCO_KEYPOINTS
    missing = int(np.sum(keypoints.sum(axis=2) == 0))
    missing_ratio = missing / total_joints
    stats["missing_joint_ratio"] = round(missing_ratio, 4)
    stats["missing_joint_count"] = missing

    # Frame-to-frame displacement stats (per joint, aggregated)
    if T > 1:
        all_diffs = []
        max_disp = 0.0
        jump_count = 0
        for t in range(1, T):
            diff = np.linalg.norm(keypoints[t] - keypoints[t-1], axis=1)
            valid = (keypoints[t].sum(axis=1) != 0) & (keypoints[t-1].sum(axis=1) != 0)
            if valid.sum() > 0:
                all_diffs.extend(diff[valid].tolist())
                max_disp = max(max_disp, float(np.max(diff[valid])))
                # Count jumps > JUMP_THRESHOLD_PX
                jump_count += int(np.sum(diff[valid] > JUMP_THRESHOLD_PX))
        stats["mean_frame_disp"] = round(float(np.mean(all_diffs)), 2) if all_diffs else 0.0
        stats["max_frame_disp"] = round(max_disp, 2)
        stats["jump_count"] = jump_count
    else:
        stats["mean_frame_disp"] = 0.0
        stats["max_frame_disp"] = 0.0
        stats["jump_count"] = 0

    # -- QC checks --
    if T < 5:
        reasons.append(f"Too few frames: {T}")
        stats["passed"] = False
        stats["reasons"] = reasons
        return False, reasons, stats

    # Check 1: Critical joint visibility across frames
    critical_vis_ratios = {}
    for coco_idx in CRITICAL_COCO_INDICES:
        visible_frames = np.sum(visibilities[:, coco_idx] >= CONFIDENCE_THRESHOLD)
        ratio = visible_frames / T
        critical_vis_ratios[f"critical_kpt_{coco_idx}_vis_ratio"] = round(ratio, 4)
        if ratio < 0.5:
            reasons.append(
                f"Critical joint {coco_idx} visible only {ratio:.0%} of frames"
            )
    stats.update(critical_vis_ratios)

    # Check 2: Relative jump detection
    OUTLIER_RATIO = 3.0
    MIN_ABSOLUTE_PX = 150
    JUMP_EXCLUDE_JOINTS = {9, 10, 15, 16}
    outlier_jump_count = 0
    for t in range(1, T):
        diff = np.linalg.norm(keypoints[t] - keypoints[t-1], axis=1)
        valid = (keypoints[t].sum(axis=1) != 0) & (keypoints[t-1].sum(axis=1) != 0)
        if valid.sum() < 3:
            continue

        valid_diffs = diff[valid]
        median_disp = np.median(valid_diffs)

        for j_idx in np.where(valid)[0]:
            if j_idx in JUMP_EXCLUDE_JOINTS:
                continue
            joint_disp = diff[j_idx]
            if joint_disp > MIN_ABSOLUTE_PX and median_disp > 0 and (joint_disp / median_disp) > OUTLIER_RATIO:
                outlier_jump_count += 1
                reasons.append(
                    f"Outlier joint {j_idx} at frame {t}: "
                    f"{joint_disp:.0f}px vs median {median_disp:.0f}px "
                    f"(ratio {joint_disp/median_disp:.1f}x)"
                )
    stats["outlier_jump_count"] = outlier_jump_count

    # Check 3: Overall missing joint ratio
    if missing_ratio > MISSING_JOINT_RATIO_LIMIT:
        reasons.append(f"Missing joint ratio: {missing_ratio:.0%} (limit: {MISSING_JOINT_RATIO_LIMIT:.0%})")

    passed = len(reasons) == 0
    stats["passed"] = passed
    stats["reasons"] = reasons
    return passed, reasons, stats


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


def load_mapping(mapping_path: Optional[str]) -> Dict[str, str]:
    """Load folder-name -> stroke-type mapping from JSON file."""
    if mapping_path is None:
        return {}
    with open(mapping_path, 'r') as f:
        raw = json.load(f)
    # Normalize keys: strip, lowercase; skip metadata keys starting with _
    return {k.strip().lower(): v.strip().lower()
            for k, v in raw.items() if not k.startswith('_')}


def process_clips_directory(
    clips_dir: str,
    output_dir: str,
    flagged_dir: str = "data/flagged",
    start_id: int = 1000,
    verify: bool = False,
    stats_output: Optional[str] = None,
    mapping: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Process all video clips in a directory structure:
      clips_dir/{stroke_type}/*.mp4 -> output_dir/{stroke_type}_v2.csv

    If --mapping is provided, folder names are translated to stroke types
    using the JSON mapping file. Unmapped folders are skipped.

    Flagged clips (QC failures) are copied to flagged_dir with a log.

    Args:
        clips_dir: Root directory containing stroke_type subdirectories
        output_dir: Output directory for CSVs (e.g. data/youtube/)
        flagged_dir: Directory for QC-failed clips
        start_id: Starting sample ID for new data
        verify: If True, print summary stats per clip
        stats_output: If set, path to save structured QC stats JSON
        mapping: Optional path to JSON file mapping folder->stroke_type

    Returns:
        List of per-clip QC stats dicts
    """
    clips_dir = Path(clips_dir)
    output_dir = Path(output_dir)
    flagged_dir = Path(flagged_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    flagged_dir.mkdir(parents=True, exist_ok=True)

    folder_to_stroke = load_mapping(mapping)
    use_mapping = len(folder_to_stroke) > 0
    if use_mapping:
        print(f"Loaded mapping: {len(folder_to_stroke)} folder->stroke entries")
        valid_strokes = set(folder_to_stroke.values())
        print(f"  Target stroke types: {sorted(valid_strokes)}")

    current_id = start_id
    total_passed = 0
    total_flagged = 0
    total_skipped = 0
    all_stats: List[Dict[str, Any]] = []

    # Open a flagged log
    flag_log_path = flagged_dir / "qc_log.txt"
    flag_log = open(flag_log_path, 'a')

    for stroke_dir in sorted(clips_dir.iterdir()):
        if not stroke_dir.is_dir():
            continue

        raw_folder_name = stroke_dir.name
        if use_mapping:
            folder_key = raw_folder_name.strip().lower()
            if folder_key not in folder_to_stroke:
                print(f"  Skipping unmapped: {raw_folder_name}")
                total_skipped += 1
                continue
            stroke_type = folder_to_stroke[folder_key]
        else:
            stroke_type = raw_folder_name.lower()

        video_files = sorted(
            [f for f in stroke_dir.iterdir() if f.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv')]
        )

        if not video_files:
            continue

        label = f"{raw_folder_name} -> {stroke_type}" if use_mapping else stroke_type
        print(f"\n-- [{label}] {len(video_files)} clips --")

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
            print(f"  -> {vf.name}...", end=" ", flush=True)
            try:
                keypoints, visibilities, fps = extract_keypoints_from_video(str(vf))
            except Exception as e:
                print(f"ERROR: {e}")
                continue

            passed, reasons, stats = quality_control(keypoints, visibilities, vf.name)
            stats["sample_id"] = current_id if passed else -1
            stats["stroke_type"] = stroke_type
            stats["fps"] = fps
            all_stats.append(stats)

            if passed:
                rows = keypoints_to_csv_rows(keypoints, current_id, stroke_type)
                all_rows.extend(rows)
                print(f"OK ({keypoints.shape[0]} frames, id={current_id})")
                current_id += 1
                total_passed += 1
            else:
                # Move to flagged directory
                flagged_stroke_dir = flagged_dir / stroke_type
                flagged_stroke_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(vf, flagged_stroke_dir / vf.name)

                reason_str = "; ".join(reasons)
                flag_log.write(f"{vf.name}: {reason_str}\n")
                print(f"!! FLAGGED ({reason_str})")
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

    # -- Save structured QC stats --
    if stats_output and all_stats:
        stats_path = Path(stats_output)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, 'w') as f:
            json.dump(all_stats, f, indent=2)
        print(f"\n  QC stats saved to {stats_path}")

    # -- Print aggregate summary --
    print(f"\n{'='*50}")
    print(f"  Passed QC: {total_passed}")
    print(f"  Flagged:   {total_flagged} (see {flag_log_path})")
    print(f"{'='*50}")

    # Per-stroke summary
    if all_stats:
        print(f"\n{'-'*60}")
        print(f"  Per-Stroke QC Summary")
        print(f"{'-'*60}")
        from collections import defaultdict
        stroke_stats = defaultdict(lambda: {"total": 0, "passed": 0, "flagged": 0,
                                            "mean_vis": [], "missing_ratio": [],
                                            "mean_disp": [], "jump_count": []})
        for s in all_stats:
            st = s["stroke_type"]
            stroke_stats[st]["total"] += 1
            if s["passed"]:
                stroke_stats[st]["passed"] += 1
            else:
                stroke_stats[st]["flagged"] += 1
            stroke_stats[st]["mean_vis"].append(s.get("kpt_9_mean_vis", 0))  # right wrist
            stroke_stats[st]["missing_ratio"].append(s["missing_joint_ratio"])
            stroke_stats[st]["mean_disp"].append(s["mean_frame_disp"])
            stroke_stats[st]["jump_count"].append(s["jump_count"])

        for st in sorted(stroke_stats.keys()):
            ss = stroke_stats[st]
            pass_rate = ss["passed"] / ss["total"] * 100 if ss["total"] else 0
            avg_wrist_vis = np.mean(ss["mean_vis"]) if ss["mean_vis"] else 0
            avg_missing = np.mean(ss["missing_ratio"]) * 100 if ss["missing_ratio"] else 0
            avg_disp = np.mean(ss["mean_disp"]) if ss["mean_disp"] else 0
            avg_jumps = np.mean(ss["jump_count"]) if ss["jump_count"] else 0
            print(f"  {st:<12} | {ss['total']:>4} clips | {pass_rate:5.1f}% pass | "
                  f"wrist_vis={avg_wrist_vis:.2f} | missing={avg_missing:.1f}% | "
                  f"mean_disp={avg_disp:.0f}px | jumps={avg_jumps:.1f}")

    return all_stats


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
    parser.add_argument(
        "--stats", type=str, default="data/qc_stats.json",
        help="Output path for structured QC stats JSON (default: data/qc_stats.json)"
    )
    parser.add_argument(
        "--mapping", type=str, default=None,
        help="JSON file mapping folder names -> stroke types (for non-standard layouts)"
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
        stats_output=args.stats,
        mapping=args.mapping,
    )
