"""
tools/review_clips.py
Batch review tool for flagged clips.
Keyboard: Y=approve, N=reject, S=skip, R=replay, Q=quit
"""
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from pathlib import Path
import argparse
import shutil
import json

POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
    (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),(11,23),
    (12,24),(23,24),(23,25),(24,26),(25,27),(26,28),(27,29),
    (28,30),(29,31),(30,32),(27,31),(28,32)
]

PROGRESS_FILE = "tools/.review_progress.json"


def load_progress() -> dict:
    if Path(PROGRESS_FILE).exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {}


def save_progress(progress: dict):
    Path(PROGRESS_FILE).parent.mkdir(exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def play_clip(cap, landmarker, clip_path: Path, clip_idx: int, total: int) -> str:
    """
    Play a single clip with skeleton overlay.
    Returns: 'approve', 'reject', 'skip', 'replay', 'quit'
    """
    fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    action = None

    while cap.isOpened() and action is None:
        ret, frame = cap.read()

        # Hết video → dừng và chờ quyết định
        if not ret:
            cv2.putText(frame if ret else
                        cv2.imread(str(clip_path)),  # fallback
                        "END — Y:Approve  N:Reject  S:Skip  R:Replay",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            # Chờ input sau khi video hết
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('y'): return 'approve'
                if key == ord('n'): return 'reject'
                if key == ord('s'): return 'skip'
                if key == ord('r'): return 'replay'
                if key == ord('q'): return 'quit'
            break

        h, w, _ = frame.shape

        # Skeleton overlay
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(frame_idx * 1000 / fps)
        frame_idx += 1

        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks[0]
            for s, e in POSE_CONNECTIONS:
                p1, p2 = landmarks[s], landmarks[e]
                if p1.visibility > 0.3 and p2.visibility > 0.3:
                    cv2.line(frame,
                             (int(p1.x*w), int(p1.y*h)),
                             (int(p2.x*w), int(p2.y*h)),
                             (0, 0, 255), 2)
            for lm in landmarks:
                if lm.visibility > 0.3:
                    cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 3, (0,255,0), -1)

        # HUD
        current_sec = frame_idx / fps
        stroke_type = clip_path.parent.name

        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}  |  {current_sec:.2f}s",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"[{clip_idx+1}/{total}] {clip_path.name}  ({stroke_type})",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.putText(frame, "Y:Approve  N:Reject  S:Skip  R:Replay  Q:Quit",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

        cv2.imshow("BSQAv2 - Review Mode", frame)

        key = cv2.waitKey(30) & 0xFF  # ~30fps playback
        if key == ord('y'): action = 'approve'
        elif key == ord('n'): action = 'reject'
        elif key == ord('s'): action = 'skip'
        elif key == ord('r'): action = 'replay'
        elif key == ord('q'): action = 'quit'
        elif key == ord(' '): cv2.waitKey(-1)  # pause

    return action or 'skip'


def review_folder(folder: str, approved_dir: str, rejected_dir: str):
    folder = Path(folder)
    stroke_type = folder.name

    approved_out = Path(approved_dir) / stroke_type
    rejected_out = Path(rejected_dir) / stroke_type
    approved_out.mkdir(parents=True, exist_ok=True)
    rejected_out.mkdir(parents=True, exist_ok=True)

    clips = sorted(folder.glob("*.mp4"))
    if not clips:
        print(f"Không có clip nào trong {folder}")
        return

    # Load tiến độ đã review dở
    progress = load_progress()
    folder_key = str(folder.resolve())
    reviewed = set(progress.get(folder_key, []))
    pending = [c for c in clips if c.name not in reviewed]

    print(f"\n{'='*50}")
    print(f"Folder: {folder}")
    print(f"Total: {len(clips)} | Reviewed: {len(reviewed)} | Pending: {len(pending)}")
    print(f"Controls: Y=Approve  N=Reject  S=Skip  R=Replay  SPACE=Pause  Q=Quit")
    print(f"{'='*50}\n")

    # Setup MediaPipe
    # model_path = Path("BSQAv2/pose_landmarker_full.task")
    model_path = Path(__file__).resolve().parent / "pose_landmarker_full.task"
    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )

    approved_count = rejected_count = skipped_count = 0

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        i = 0
        while i < len(pending):
            clip_path = pending[i]

            cap = cv2.VideoCapture(str(clip_path))
            if not cap.isOpened():
                print(f"⚠ Cannot open: {clip_path.name}")
                i += 1
                continue

            while True:  # replay loop
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                action = play_clip(cap, landmarker, clip_path, i, len(pending))

                if action == 'replay':
                    continue  # rewind và play lại
                break

            cap.release()

            if action == 'approve':
                shutil.move(str(clip_path), approved_out / clip_path.name)
                print(f"  ✓ Approved → {approved_out.name}/{clip_path.name}")
                approved_count += 1
                reviewed.add(clip_path.name)
            elif action == 'reject':
                shutil.move(str(clip_path), rejected_out / clip_path.name)
                print(f"  ✗ Rejected → {rejected_out.name}/{clip_path.name}")
                rejected_count += 1
                reviewed.add(clip_path.name)
            elif action == 'skip':
                print(f"  ~ Skipped: {clip_path.name}")
                skipped_count += 1
            elif action == 'quit':
                print("\nĐã lưu tiến độ. Chạy lại để tiếp tục.")
                break

            # Lưu progress sau mỗi action
            progress[folder_key] = list(reviewed)
            save_progress(progress)

            i += 1

    cv2.destroyAllWindows()

    print(f"\n{'='*50}")
    print(f"Session kết thúc:")
    print(f"  Approved : {approved_count}")
    print(f"  Rejected : {rejected_count}")
    print(f"  Skipped  : {skipped_count}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch review flagged clips")
    parser.add_argument("--folder", required=True,
                        help="Folder chứa clips cần review (vd: data/flagged/smash)")
    parser.add_argument("--approved-dir", default="data/clips",
                        help="Output dir cho approved clips")
    parser.add_argument("--rejected-dir", default="data/rejected",
                        help="Output dir cho rejected clips")
    args = parser.parse_args()

    review_folder(args.folder, args.approved_dir, args.rejected_dir)