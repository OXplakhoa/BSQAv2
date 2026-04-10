"""
Batch video trimmer using ffmpeg.
Reads timestamp_log.csv and trims raw videos into individual stroke clips.
Output: data/clips/{stroke_type}/{video_id}_{index}.mp4
"""
import argparse
import csv
import subprocess
from pathlib import Path
from collections import defaultdict


def time_to_seconds(time_str: str) -> float:
    """Convert HH:MM:SS or MM:SS to seconds."""
    parts = time_str.strip().split(':')
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    else:
        return float(parts[0])


def trim_clips(
    log_path: str,
    raw_video_dir: str,
    output_dir: str,
    dry_run: bool = False,
) -> None:
    """
    Trim raw videos into individual stroke clips.

    Args:
        log_path: Path to timestamp_log.csv
        raw_video_dir: Directory containing downloaded raw videos
        output_dir: Output directory for clips (e.g. data/clips/)
        dry_run: If True, only print what would be trimmed
    """
    raw_video_dir = Path(raw_video_dir)
    output_dir = Path(output_dir)

    with open(log_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Track per-video clip index for unique naming
    clip_counter = defaultdict(int)
    trimmed = 0
    skipped = 0
    failed = 0

    for row in rows:
        if not row.get('video_url') or not row.get('stroke_type'):
            continue
            
        url = row['video_url'].strip()
        stroke_type = row['stroke_type'].strip().lower()
        start_time = row['start_time'].strip()
        end_time = row['end_time'].strip()

        # Find the raw video file
        import sys
        
        # Thêm thư mục gốc BSQAv2 vào path để Python luôn tìm thấy folder tools/
        project_root = Path(__file__).resolve().parent.parent
        if str(project_root) not in sys.path:
            sys.path.append(str(project_root))
            
        from tools.download_clips import extract_video_id
        try:
            vid_id = extract_video_id(url)
        except ValueError as e:
            print(f"⚠ Skipping: {e}")
            skipped += 1
            continue

        raw_files = list(raw_video_dir.glob(f"{vid_id}.*"))
        if not raw_files:
            print(f"⚠ Raw video not found for {vid_id} — run download_clips.py first")
            skipped += 1
            continue

        raw_path = raw_files[0]
        clip_counter[vid_id] += 1
        clip_idx = clip_counter[vid_id]

        # Create output directory
        stroke_dir = output_dir / stroke_type
        stroke_dir.mkdir(parents=True, exist_ok=True)

        clip_name = f"{vid_id}_{clip_idx:03d}.mp4"
        clip_path = stroke_dir / clip_name

        if clip_path.exists():
            print(f"✓ Exists: {clip_path.name}")
            skipped += 1
            continue

        # Calculate duration
        start_sec = time_to_seconds(start_time)
        end_sec = time_to_seconds(end_time)
        duration = end_sec - start_sec

        if duration <= 0:
            print(f"⚠ Invalid time range for {vid_id} clip {clip_idx}: {start_time} → {end_time}")
            skipped += 1
            continue

        if dry_run:
            print(f"  [DRY RUN] {raw_path.name} [{start_time} → {end_time}] → {clip_path.name}")
            continue

        # Trim with ffmpeg (re-encode for accurate cuts)
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_sec),
            "-i", str(raw_path),
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "fast",
            "-an",  # No audio needed for pose estimation
            str(clip_path),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"  ✓ Trimmed: {clip_path.name} ({duration:.1f}s)")
            trimmed += 1
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed: {clip_path.name} — {e.stderr[:200]}")
            failed += 1
        except FileNotFoundError:
            print("  ✗ ffmpeg not found. Install with: brew install ffmpeg")
            return

    print(f"\n{'='*50}")
    print(f"Trimmed: {trimmed} | Skipped: {skipped} | Failed: {failed}")
    print(f"{'='*50}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trim raw videos into individual stroke clips"
    )
    parser.add_argument(
        "--log", type=str, default="tools/timestamp_log.csv",
        help="Path to timestamp_log.csv"
    )
    parser.add_argument(
        "--raw-dir", type=str, default="data/raw_videos",
        help="Directory containing raw downloaded videos"
    )
    parser.add_argument(
        "--output", type=str, default="data/clips",
        help="Output directory for trimmed clips"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only print what would be trimmed"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trim_clips(args.log, args.raw_dir, args.output, dry_run=args.dry_run)
