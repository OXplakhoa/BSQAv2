"""
Batch video downloader using yt-dlp.
Reads timestamp_log.csv and downloads unique videos to data/raw_videos/.
Caches downloads by video_id — skips already-downloaded videos.
"""
import argparse
import csv
import subprocess
import re
from pathlib import Path


def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from URL."""
    patterns = [
        r'(?:v=|/)([0-9A-Za-z_-]{11})(?:\?|&|$|#)',
        r'youtu\.be/([0-9A-Za-z_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Cannot extract video ID from: {url}")


def download_videos(log_path: str, output_dir: str, dry_run: bool = False) -> None:
    """
    Download unique videos from timestamp log.

    Args:
        log_path: Path to timestamp_log.csv
        output_dir: Directory to save raw videos (e.g. data/raw_videos/)
        dry_run: If True, only print what would be downloaded
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(log_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Deduplicate by video_id
    seen_ids = set()
    unique_urls = []
    for row in rows:
        url = row['video_url'].strip()
        try:
            vid_id = extract_video_id(url)
        except ValueError as e:
            print(f"⚠ Skipping invalid URL: {e}")
            continue

        if vid_id in seen_ids:
            continue
        seen_ids.add(vid_id)

        # Cache check: skip if already downloaded
        existing = list(output_dir.glob(f"{vid_id}.*"))
        if existing:
            print(f"✓ Cached: {vid_id} ({existing[0].name})")
            continue

        unique_urls.append((vid_id, url))

    print(f"\n{'='*50}")
    print(f"Total unique videos: {len(seen_ids)}")
    print(f"Already cached: {len(seen_ids) - len(unique_urls)}")
    print(f"To download: {len(unique_urls)}")
    print(f"{'='*50}\n")

    if dry_run:
        for vid_id, url in unique_urls:
            print(f"  [DRY RUN] Would download: {vid_id} from {url}")
        return

    failed = []
    for i, (vid_id, url) in enumerate(unique_urls, 1):
        print(f"[{i}/{len(unique_urls)}] Downloading {vid_id}...")
        cmd = [
            "yt-dlp",
            "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]",
            "--merge-output-format", "mp4",
            "-o", str(output_dir / f"{vid_id}.%(ext)s"),
            "--no-playlist",
            url,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"  ✓ Done: {vid_id}")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed: {vid_id} — {e.stderr[:200]}")
            failed.append((vid_id, url, str(e.stderr[:200])))
        except FileNotFoundError:
            print("  ✗ yt-dlp not found. Install with: pip install yt-dlp")
            return

    if failed:
        print(f"\n⚠ {len(failed)} downloads failed:")
        for vid_id, url, err in failed:
            print(f"  - {vid_id}: {err}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download YouTube videos from timestamp log"
    )
    parser.add_argument(
        "--log", type=str, default="tools/timestamp_log.csv",
        help="Path to timestamp_log.csv"
    )
    parser.add_argument(
        "--output", type=str, default="data/raw_videos",
        help="Output directory for raw videos"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only print what would be downloaded, don't actually download"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_videos(args.log, args.output, dry_run=args.dry_run)
