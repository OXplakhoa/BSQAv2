"""Diagnostics text helpers for demo-safe explanations."""
from __future__ import annotations

from typing import Dict, Optional

from .schema import PredictionResult


def confidence_margin(probabilities: Dict[str, float]) -> Optional[float]:
    """Return top-1 minus top-2 probability margin, or None if unavailable."""
    if len(probabilities) < 2:
        return None
    ranked = sorted(probabilities.values(), reverse=True)
    return float(ranked[0] - ranked[1])


def reliability_label(score: Optional[float]) -> str:
    """Map a 0..1 pose reliability score to a presentation label."""
    if score is None:
        return "unknown"
    if score >= 0.8:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def summarize_prediction(
    prediction: PredictionResult,
    ground_truth: Optional[str] = None,
    pose_reliability: Optional[float] = None,
    branch_name: str = "Model",
) -> str:
    """Produce academically honest prediction wording.

    If ground_truth is missing, no correctness claim is made.
    """
    label = prediction.label or "unknown"
    confidence = prediction.confidence
    confidence_text = "unknown confidence"
    if confidence is not None:
        confidence_text = f"{confidence * 100:.0f}% confidence"

    reliability = reliability_label(pose_reliability)

    if ground_truth:
        correctness = "correct" if label == ground_truth else "incorrect"
        return (
            f"{branch_name} predicted {label} with {confidence_text}. "
            f"Ground truth is {ground_truth}, so this prediction is {correctness}. "
            f"Pose reliability is {reliability}."
        )

    return (
        f"{branch_name} predicted {label} with {confidence_text}. "
        f"Pose reliability is {reliability}. This should be interpreted cautiously."
    )


def compare_branches(dl: PredictionResult, rf: PredictionResult) -> str:
    """Explain DL/RF agreement or disagreement."""
    if not dl.label or not rf.label:
        return "DL/RF agreement cannot be computed because one prediction is missing."
    if dl.label == rf.label:
        return (
            f"Both branches predict {dl.label}. Agreement increases confidence, "
            "but the result still depends on pose quality and class ambiguity."
        )
    return (
        f"The branches disagree: DL predicts {dl.label}, while RF predicts {rf.label}. "
        "This is useful for error analysis because engineered biomechanical "
        "features and temporal deep learning may emphasize different evidence."
    )
