"""Candidate scan tool for Pipeline Observatory curated sample selection.

The tool samples reviewed clips from data/clips, optionally runs MediaPipe and the
canonical skeleton pipeline, and writes a JSON candidate report for final manual
selection. Use --dry-run to inspect the balanced candidate list without doing
slow pose extraction.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.config import SEED, STROKE_TYPES
from src.observatory.artifacts import ArtifactRegistry
from src.observatory.pipeline import run_skeleton_pipeline

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def find_video_candidates(clips_dir: Path) -> List[Dict[str, str]]:
    """Return candidate video records from clips_dir/{stroke_type}/*.video."""
    clips_dir = Path(clips_dir)
    candidates: List[Dict[str, str]] = []
    for stroke in STROKE_TYPES:
        stroke_dir = clips_dir / stroke
        if not stroke_dir.exists():
            continue
        for video_path in sorted(stroke_dir.iterdir()):
            if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            candidates.append(
                {
                    "sample_id": f"{stroke}_{video_path.stem}",
                    "stroke_type": stroke,
                    "video_path": video_path.as_posix(),
                }
            )
    return candidates


def select_candidate_videos(
    candidates: List[Dict[str, str]],
    max_total: int = 35,
    limit_per_class: Optional[int] = None,
    seed: int = SEED,
) -> List[Dict[str, str]]:
    """Select a deterministic, class-balanced candidate subset."""
    by_class: Dict[str, List[Dict[str, str]]] = {stroke: [] for stroke in STROKE_TYPES}
    for item in candidates:
        by_class.setdefault(item["stroke_type"], []).append(item)

    rng = random.Random(seed)
    if limit_per_class is None:
        active_classes = [stroke for stroke, rows in by_class.items() if rows]
        limit_per_class = max(1, (max_total + len(active_classes) - 1) // len(active_classes))

    selected: List[Dict[str, str]] = []
    for stroke in STROKE_TYPES:
        rows = list(by_class.get(stroke, []))
        rng.shuffle(rows)
        selected.extend(rows[:limit_per_class])

    selected = selected[:max_total]
    selected.sort(key=lambda item: (item["stroke_type"], item["video_path"]))
    return selected


def scan_candidate(
    candidate: Dict[str, str],
    registry: ArtifactRegistry,
    rf_bundle_path: Path,
    dl_checkpoint_path: Optional[Path] = None,
) -> Dict:
    """Run MediaPipe + canonical pipeline for one candidate clip."""
    from src.utils.video_to_csv import extract_keypoints_from_video

    keypoints, visibilities, fps = extract_keypoints_from_video(candidate["video_path"])
    run = run_skeleton_pipeline(
        sample_id=candidate["sample_id"],
        raw_keypoints=keypoints,
        visibilities=visibilities,
        source_video_path=candidate["video_path"],
        ground_truth=candidate["stroke_type"],
        rf_bundle_path=rf_bundle_path,
        dl_checkpoint_path=dl_checkpoint_path,
    )
    run.video_metadata["fps"] = int(fps)
    run.video_metadata["stroke_type"] = candidate["stroke_type"]
    registry.save_pipeline_run(run)

    rf_correct = run.rf_prediction.label == candidate["stroke_type"]
    dl_correct = None
    if run.dl_prediction.label is not None:
        dl_correct = run.dl_prediction.label == candidate["stroke_type"]

    return {
        **candidate,
        "run_id": run.run_id,
        "pipeline_run_dir": registry.pipeline_run_dir(run.run_id).as_posix(),
        "frame_count": int(keypoints.shape[0]),
        "fps": int(fps),
        "pose_reliability_score": run.pose_qc.get("reliability_score"),
        "pose_reliability_label": run.pose_qc.get("reliability_label"),
        "pose_warnings": run.pose_qc.get("warnings", []),
        "rf_prediction": run.rf_prediction.label,
        "rf_confidence": run.rf_prediction.confidence,
        "rf_correct": rf_correct,
        "dl_prediction": run.dl_prediction.label,
        "dl_confidence": run.dl_prediction.confidence,
        "dl_correct": dl_correct,
        "rf_dl_agree": (
            run.rf_prediction.label == run.dl_prediction.label
            if run.rf_prediction.label and run.dl_prediction.label else None
        ),
    }


def write_report(report: Dict, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Scan candidate clips for curated Streamlit cases")
    parser.add_argument("--clips-dir", type=str, default="data/clips")
    parser.add_argument("--output", type=str, default=None,
                        help="Candidate report JSON path")
    parser.add_argument("--max-total", type=int, default=35)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--rf-bundle", type=str,
                        default="webapp/artifacts/models/rf_baseline/rf_model_bundle.joblib")
    parser.add_argument("--dl-checkpoint", type=str, default=None,
                        help="Optional GCN+BiLSTM+Attention checkpoint for DL predictions")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only write selected candidate list; skip MediaPipe/pipeline")
    return parser.parse_args()


def main():
    args = parse_args()
    registry = ArtifactRegistry()
    registry.ensure_layout()

    candidates = find_video_candidates(Path(args.clips_dir))
    selected = select_candidate_videos(
        candidates,
        max_total=args.max_total,
        limit_per_class=args.limit_per_class,
        seed=SEED,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output) if args.output else (
        registry.curated_dir / f"candidate_scan_{timestamp}.json"
    )

    report = {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(),
        "clips_dir": args.clips_dir,
        "dry_run": bool(args.dry_run),
        "total_available": len(candidates),
        "total_selected": len(selected),
        "selected": selected,
        "results": [],
    }

    if not args.dry_run:
        rf_bundle_path = Path(args.rf_bundle)
        dl_checkpoint_path = Path(args.dl_checkpoint) if args.dl_checkpoint else None
        for idx, candidate in enumerate(selected, start=1):
            print(f"[{idx}/{len(selected)}] {candidate['video_path']}", flush=True)
            try:
                report["results"].append(
                    scan_candidate(candidate, registry, rf_bundle_path, dl_checkpoint_path)
                )
            except Exception as exc:
                report["results"].append({**candidate, "error": str(exc)})
                print(f"  ERROR: {exc}", flush=True)

    write_report(report, output_path)
    print(f"Candidate report saved: {output_path}")


if __name__ == "__main__":
    main()
