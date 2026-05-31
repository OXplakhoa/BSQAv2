"""Build curated sample manifest from a candidate scan report.

The manifest is the stable case bank used by Streamlit. This tool does not run
MediaPipe or inference; it ranks existing candidate report rows and writes about
12 presentation-ready samples with teaching labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.config import STROKE_TYPES


Result = Dict


def load_candidate_report(path: Path) -> Dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Candidate report not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def latest_candidate_report(curated_dir: Path = Path("webapp/artifacts/curated")) -> Path:
    reports = sorted(Path(curated_dir).glob("candidate_scan_*.json"))
    if not reports:
        raise FileNotFoundError(f"No candidate_scan_*.json reports found in {curated_dir}")
    return reports[-1]


def _score(row: Result) -> float:
    pose = float(row.get("pose_reliability_score") or 0.0)
    rf_conf = float(row.get("rf_confidence") or 0.0)
    dl_conf = float(row.get("dl_confidence") or 0.0)
    dl_bonus = 0.15 if row.get("dl_correct") is True else 0.0
    warning_penalty = 0.10 if row.get("pose_warnings") else 0.0
    return pose + (0.4 * rf_conf) + (0.2 * dl_conf) + dl_bonus - warning_penalty


def _low_conf_score(row: Result) -> float:
    return float(row.get("rf_confidence") or 1.0)


def _disagreement_score(row: Result) -> float:
    pose = float(row.get("pose_reliability_score") or 0.0)
    dl_conf = float(row.get("dl_confidence") or 0.0)
    warning_bonus = 0.2 if row.get("pose_warnings") else 0.0
    return pose + dl_conf + warning_bonus


def _title(label: str, row: Result) -> str:
    return f"{label}: {row['stroke_type'].replace('_', ' ').title()}"


def _diagnosis(row: Result) -> str:
    pieces = []
    if row.get("rf_correct") is True:
        pieces.append(f"Random Forest correctly predicts {row.get('rf_prediction')}.")
    elif row.get("rf_prediction"):
        pieces.append(f"Random Forest predicts {row.get('rf_prediction')} instead of {row.get('stroke_type')}.")

    if row.get("dl_prediction"):
        if row.get("dl_correct") is True:
            pieces.append(f"Deep Learning also predicts {row.get('dl_prediction')} correctly.")
        else:
            pieces.append(f"Deep Learning predicts {row.get('dl_prediction')}, creating a useful comparison case.")

    if row.get("pose_warnings"):
        pieces.append("Pose warnings: " + "; ".join(row.get("pose_warnings", [])[:2]) + ".")
    return " ".join(pieces)


def _teaching_point(bucket: str, row: Result) -> str:
    if bucket.startswith("class_representative"):
        return "Clean class representative for walking through the full pipeline."
    if bucket == "both_models_correct":
        return "Agreement case where engineered features and temporal DL support the same label."
    if bucket == "rf_correct_dl_wrong":
        return "RF succeeds while DL fails, useful for explaining model disagreement."
    if bucket == "pose_warning":
        return "Pose quality issue shows how upstream tracking affects downstream predictions."
    if bucket == "low_rf_confidence":
        return "Correct but low-confidence RF case for discussing class ambiguity."
    if bucket == "lift_confusion":
        return "Lift ambiguity case, aligned with the weak DL lift performance story."
    return "Curated case for pipeline explanation and diagnostics."


def _pipeline_run_dir(row: Result) -> Optional[str]:
    value = row.get("pipeline_run_dir")
    if not value:
        return None
    normalized = str(value).replace("\\", "/")
    marker = "webapp/artifacts/pipeline_runs/"
    idx = normalized.lower().find(marker.lower())
    if idx >= 0:
        return normalized[idx:]
    return normalized


def result_to_manifest_entry(row: Result, bucket: str, title_label: str) -> Dict:
    tags = [bucket]
    if row.get("rf_correct") is True:
        tags.append("rf_correct")
    if row.get("dl_correct") is True:
        tags.append("dl_correct")
    if row.get("rf_dl_agree") is False:
        tags.append("rf_dl_disagreement")
    if row.get("pose_warnings"):
        tags.append("pose_warning")
    tags = list(dict.fromkeys(tags))

    return {
        "sample_id": row["sample_id"],
        "title": _title(title_label, row),
        "stroke_type": row["stroke_type"],
        "ground_truth": row["stroke_type"],
        "video_path": row["video_path"],
        "pipeline_run_dir": _pipeline_run_dir(row),
        "manual_review_status": "reviewed",
        "teaching_point": _teaching_point(bucket, row),
        "diagnosis": _diagnosis(row),
        "tags": tags,
        "rf_prediction": row.get("rf_prediction"),
        "rf_confidence": row.get("rf_confidence"),
        "rf_correct": row.get("rf_correct"),
        "dl_prediction": row.get("dl_prediction"),
        "dl_confidence": row.get("dl_confidence"),
        "dl_correct": row.get("dl_correct"),
        "rf_dl_agree": row.get("rf_dl_agree"),
        "pose_reliability_score": row.get("pose_reliability_score"),
        "pose_reliability_label": row.get("pose_reliability_label"),
        "pose_warnings": row.get("pose_warnings", []),
    }


def _pick_best(
    rows: Iterable[Result],
    predicate: Callable[[Result], bool],
    used: set,
    key: Callable[[Result], float] = _score,
) -> Optional[Result]:
    candidates = [row for row in rows if row.get("sample_id") not in used and predicate(row)]
    if not candidates:
        return None
    candidates.sort(key=key, reverse=True)
    return candidates[0]


def choose_curated_samples(results: List[Result], target_count: int = 12) -> List[Dict]:
    """Choose a deterministic, teaching-oriented curated sample set."""
    rows = [row for row in results if "error" not in row]
    selected: List[Dict] = []
    used = set()

    def add(row: Optional[Result], bucket: str, label: str) -> None:
        if row is None or row.get("sample_id") in used:
            return
        selected.append(result_to_manifest_entry(row, bucket, label))
        used.add(row["sample_id"])

    # One representative per class, preferring RF-correct and both-model-correct cases.
    for stroke in STROKE_TYPES:
        add(
            _pick_best(
                rows,
                lambda r, stroke=stroke: r.get("stroke_type") == stroke and r.get("rf_correct") is True,
                used,
                key=lambda r: _score(r) + (2.0 if r.get("dl_correct") is True else 0.0),
            ),
            f"class_representative_{stroke}",
            "Class representative",
        )

    # Additional high-quality agreement cases.
    for _ in range(2):
        add(
            _pick_best(
                rows,
                lambda r: r.get("rf_correct") is True and r.get("dl_correct") is True,
                used,
            ),
            "both_models_correct",
            "Both models correct",
        )

    # Disagreement/error-analysis cases.
    for _ in range(2):
        add(
            _pick_best(
                rows,
                lambda r: r.get("rf_correct") is True and r.get("dl_correct") is False,
                used,
                key=_disagreement_score,
            ),
            "rf_correct_dl_wrong",
            "RF correct, DL wrong",
        )

    # Pose warning cases.
    for _ in range(2):
        add(
            _pick_best(
                rows,
                lambda r: r.get("rf_correct") is True and bool(r.get("pose_warnings")),
                used,
                key=lambda r: -(float(r.get("pose_reliability_score") or 1.0)),
            ),
            "pose_warning",
            "Pose warning",
        )

    # Low-confidence RF case.
    add(
        _pick_best(
            rows,
            lambda r: r.get("rf_correct") is True,
            used,
            key=lambda r: -_low_conf_score(r),
        ),
        "low_rf_confidence",
        "Low RF confidence",
    )

    # Explicit lift ambiguity case if one remains.
    add(
        _pick_best(
            rows,
            lambda r: (
                r.get("stroke_type") == "lift"
                and r.get("rf_correct") is True
                and r.get("dl_correct") is False
            ),
            used,
            key=_disagreement_score,
        ),
        "lift_confusion",
        "Lift confusion",
    )

    # Fill any remaining slots by general score.
    while len(selected) < target_count:
        row = _pick_best(rows, lambda r: r.get("rf_correct") is True, used)
        if row is None:
            break
        add(row, "high_quality_filler", "High-quality extra")

    return selected[:target_count]


def build_manifest(report_path: Path, target_count: int = 12) -> Dict:
    report = load_candidate_report(report_path)
    samples = choose_curated_samples(report.get("results", []), target_count=target_count)
    return {
        "schema_version": 1,
        "description": "Final curated sample bank for BSQAv2 Pipeline Observatory.",
        "generated_from_report": str(report_path).replace("\\", "/"),
        "created_at": datetime.now().isoformat(),
        "selection_strategy": "automatic_teaching_buckets_v1",
        "target_count": target_count,
        "samples": samples,
    }


def write_manifest(manifest: Dict, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Build curated Streamlit sample manifest")
    parser.add_argument("--report", type=str, default=None,
                        help="Candidate scan report. Defaults to latest candidate_scan_*.json")
    parser.add_argument("--output", type=str,
                        default="webapp/artifacts/curated/manifest.json")
    parser.add_argument("--target-count", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    report_path = Path(args.report) if args.report else latest_candidate_report()
    manifest = build_manifest(report_path, target_count=args.target_count)

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    output_path = write_manifest(manifest, Path(args.output))
    print(f"Curated manifest saved: {output_path}")
    print(f"Samples: {len(manifest['samples'])}")
    for sample in manifest["samples"]:
        print(f"- {sample['sample_id']} [{', '.join(sample['tags'])}]")


if __name__ == "__main__":
    main()
