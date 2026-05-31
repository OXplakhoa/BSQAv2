"""Error-analysis table preparation helpers for Streamlit pages."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set

from .bootstrap import ensure_project_imports

ensure_project_imports()

from src.observatory.diagnostics import confidence_margin
from src.observatory.schema import CuratedSample, PipelineRun, PredictionResult


def _rounded(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def _status(prediction: PredictionResult, ground_truth: Optional[str]) -> str:
    if not ground_truth or not prediction.label:
        return "unknown"
    return "correct" if prediction.label == ground_truth else "incorrect"


def prediction_audit_rows(run: PipelineRun, ground_truth: Optional[str]) -> List[Dict[str, object]]:
    """Summarize correctness, confidence, and margin for RF and DL branches."""
    branches = [
        ("Random Forest", run.rf_prediction),
        ("Deep Learning", run.dl_prediction),
    ]
    rows: List[Dict[str, object]] = []
    for branch, prediction in branches:
        rows.append({
            "branch": branch,
            "prediction": prediction.label or "missing",
            "ground_truth": ground_truth or "unknown",
            "status": _status(prediction, ground_truth),
            "confidence": _rounded(prediction.confidence),
            "margin": _rounded(confidence_margin(prediction.probabilities)),
        })
    return rows


def probability_comparison_rows(run: PipelineRun) -> List[Dict[str, object]]:
    """Merge RF and DL probability dictionaries for disagreement inspection."""
    labels: Set[str] = set(run.rf_prediction.probabilities) | set(run.dl_prediction.probabilities)
    rows: List[Dict[str, object]] = []
    for label in sorted(labels):
        rf_probability = float(run.rf_prediction.probabilities.get(label, 0.0))
        dl_probability = float(run.dl_prediction.probabilities.get(label, 0.0))
        rows.append({
            "class": label,
            "rf_probability": rf_probability,
            "dl_probability": dl_probability,
            "absolute_delta": abs(rf_probability - dl_probability),
        })
    rows.sort(key=lambda row: row["absolute_delta"], reverse=True)
    return rows


def branch_agreement_label(run: PipelineRun) -> str:
    if not run.rf_prediction.label or not run.dl_prediction.label:
        return "missing"
    return "agree" if run.rf_prediction.label == run.dl_prediction.label else "disagree"


def pose_risk_rows(run: PipelineRun) -> List[Dict[str, object]]:
    """Expose pose reliability and warnings as risk-factor rows."""
    score = run.pose_qc.get("reliability_score")
    label = run.pose_qc.get("reliability_label") or "unknown"
    rows: List[Dict[str, object]] = [
        {
            "factor": "Pose reliability",
            "level": label,
            "detail": f"score={float(score):.3f}" if score is not None else "score unavailable",
        }
    ]
    for warning in run.pose_qc.get("warnings", []):
        rows.append({
            "factor": "Pose warning",
            "level": "warning",
            "detail": str(warning),
        })
    return rows


def known_confusion_note(ground_truth: Optional[str], predicted: Optional[str]) -> str:
    """Return short known-confusion explanation for common badminton ambiguities."""
    if not ground_truth or not predicted:
        return "No ground truth or prediction is available, so class confusion cannot be diagnosed."
    if ground_truth == predicted:
        return "Predictions agree, so no class-confusion note is needed."

    pair = {ground_truth, predicted}
    if pair == {"lift", "clear"}:
        return (
            "Lift and clear can look similar in 2D pose because both involve upward arm motion; "
            "the contact point, shuttle direction, and court context are not fully visible in skeleton-only data."
        )
    if pair == {"drop_shot", "smash"}:
        return (
            "Drop shot and smash can share similar preparation frames; the decisive difference is often "
            "late deceleration and shuttle speed, which skeleton-only features capture imperfectly."
        )
    if pair == {"net_shot", "lift"}:
        return (
            "Net shot and lift both occur near the front court; wrist/forearm motion and shuttle trajectory "
            "are subtle in the COCO-17 pose representation."
        )
    return (
        f"The model confuses {ground_truth} with {predicted}. This may reflect overlapping motion features, "
        "pose-estimation noise, or missing shuttle/racket context."
    )


def curated_error_case_rows(
    samples: Iterable[CuratedSample],
    selected_sample_id: Optional[str] = None,
) -> List[Dict[str, object]]:
    """List curated cases designed for disagreement, pose-warning, or ambiguity discussion."""
    interesting_tags = {"rf_dl_disagreement", "pose_warning", "low_rf_confidence", "lift_confusion"}
    rows: List[Dict[str, object]] = []
    for sample in samples:
        tags = set(sample.tags)
        if not tags.intersection(interesting_tags):
            continue
        rows.append({
            "sample_id": sample.sample_id,
            "title": sample.title,
            "stroke_type": sample.stroke_type,
            "tags": ", ".join(sample.tags),
            "selected": sample.sample_id == selected_sample_id,
        })
    return rows
