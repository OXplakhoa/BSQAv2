"""Heuristic biomechanics rule scorer for badminton stroke quality.

The rules are intentionally transparent and conservative. They produce useful
report/demo feedback from 2D pose features, but they are not expert-validated
coaching grades.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np

from src.config import STROKE_TYPES
from src.data.biomechanics import (
    compute_contact_height,
    compute_hip_center_speed,
    compute_joint_angle,
    compute_joint_speed,
)


RS, RE, RW = 6, 8, 10
LS, LE, LW = 5, 7, 9
RH, RK, RA = 12, 14, 16
LH, LK, LA = 11, 13, 15


@dataclass
class RuleResult:
    rule: str
    score: float
    feedback: str
    observed: float
    target: str


@dataclass
class RuleEvaluation:
    stroke_type: str
    overall_score: float
    rule_results: List[RuleResult]
    feedback: List[str]


def _clip_score(value: float) -> float:
    return float(np.clip(value, 0.0, 100.0))


def _range_score(value: float, low: float, high: float, tolerance: float) -> float:
    if not np.isfinite(value):
        return 0.0
    if low <= value <= high:
        return 100.0
    if value < low:
        return _clip_score(100.0 - ((low - value) / max(tolerance, 1e-6)) * 100.0)
    return _clip_score(100.0 - ((value - high) / max(tolerance, 1e-6)) * 100.0)


def _min_score(value: float, minimum: float, tolerance: float) -> float:
    if not np.isfinite(value):
        return 0.0
    if value >= minimum:
        return 100.0
    return _clip_score(100.0 - ((minimum - value) / max(tolerance, 1e-6)) * 100.0)


def _max_score(value: float, maximum: float, tolerance: float) -> float:
    if not np.isfinite(value):
        return 0.0
    if value <= maximum:
        return 100.0
    return _clip_score(100.0 - ((value - maximum) / max(tolerance, 1e-6)) * 100.0)


def _safe_nanmax(values: np.ndarray) -> float:
    valid = values[~np.isnan(values)]
    return float(np.max(valid)) if len(valid) else np.nan


def _safe_nanmean(values: np.ndarray) -> float:
    valid = values[~np.isnan(values)]
    return float(np.mean(valid)) if len(valid) else np.nan


def _maxframe(values: np.ndarray) -> float:
    valid = ~np.isnan(values)
    if not valid.any():
        return np.nan
    return float(np.argmax(np.where(valid, values, -np.inf))) / len(values)


def _features(keypoints: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(keypoints, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1:] != (17, 2):
        raise ValueError(f"keypoints must have shape (T, 17, 2); got {arr.shape}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    rw_speed = compute_joint_speed(arr, RW)
    lw_speed = compute_joint_speed(arr, LW)
    re_angle = compute_joint_angle(arr, RS, RE, RW)
    le_angle = compute_joint_angle(arr, LS, LE, LW)
    hip_speed = compute_hip_center_speed(arr)

    wrist_y = (arr[:, RW, 1] + arr[:, LW, 1]) / 2
    wrist_vertical_travel = float(np.nanmax(wrist_y) - np.nanmin(wrist_y)) if len(wrist_y) else np.nan
    wrist_upward_motion = float(wrist_y[0] - np.nanmin(wrist_y)) if len(wrist_y) else np.nan

    wrist_speeds = np.concatenate([rw_speed[~np.isnan(rw_speed)], lw_speed[~np.isnan(lw_speed)]])
    mean_wrist_speed = float(np.mean(wrist_speeds)) if len(wrist_speeds) else np.nan

    return {
        "contact_height": compute_contact_height(arr),
        "max_wrist_speed": max(_safe_nanmax(rw_speed), _safe_nanmax(lw_speed)),
        "mean_wrist_speed": mean_wrist_speed,
        "right_elbow_max": _safe_nanmax(re_angle),
        "left_elbow_max": _safe_nanmax(le_angle),
        "dominant_elbow_max": max(_safe_nanmax(re_angle), _safe_nanmax(le_angle)),
        "impact_frame": _maxframe(rw_speed),
        "hip_speed_mean": _safe_nanmean(hip_speed),
        "wrist_vertical_travel": wrist_vertical_travel,
        "wrist_upward_motion": wrist_upward_motion,
    }


def _rule(name: str, score: float, observed: float, target: str, good: str, bad: str) -> RuleResult:
    score = _clip_score(score)
    message = good if score >= 75 else bad
    observed_text = "unknown" if not np.isfinite(observed) else f"{observed:.3f}"
    return RuleResult(
        rule=name,
        score=score,
        observed=float(observed) if np.isfinite(observed) else np.nan,
        target=target,
        feedback=f"{name}: {message} Observed={observed_text}; target={target}.",
    )


def _smash(f: Dict[str, float]) -> List[RuleResult]:
    return [
        _rule("contact_height", _min_score(f["contact_height"], 1.0, 0.7), f["contact_height"], ">= 1.0 torso units", "contact point is high", "raise contact point for a stronger overhead strike"),
        _rule("wrist_speed", _min_score(f["max_wrist_speed"], 0.08, 0.08), f["max_wrist_speed"], ">= 0.08 normalized units/frame", "wrist acceleration is strong", "increase wrist/racket-head acceleration near impact"),
        _rule("elbow_extension", _min_score(f["dominant_elbow_max"], 145.0, 35.0), f["dominant_elbow_max"], ">= 145 degrees", "arm reaches good extension", "extend the hitting arm more through contact"),
        _rule("late_impact", _range_score(f["impact_frame"], 0.45, 0.90, 0.30), f["impact_frame"], "0.45-0.90 of sequence", "impact timing is plausible", "impact timing looks early/late; check clip trimming and swing rhythm"),
    ]


def _clear(f: Dict[str, float]) -> List[RuleResult]:
    return [
        _rule("high_contact", _min_score(f["contact_height"], 0.8, 0.7), f["contact_height"], ">= 0.8 torso units", "clear has a high contact point", "clear should contact high above the body"),
        _rule("arm_extension", _min_score(f["dominant_elbow_max"], 140.0, 35.0), f["dominant_elbow_max"], ">= 140 degrees", "arm extension supports length", "extend arm more for a deeper clear"),
        _rule("controlled_speed", _range_score(f["max_wrist_speed"], 0.04, 0.22, 0.12), f["max_wrist_speed"], "0.04-0.22", "swing speed is controlled", "speed is unusually low/high for a controlled clear"),
        _rule("late_swing", _range_score(f["impact_frame"], 0.35, 0.90, 0.30), f["impact_frame"], "0.35-0.90", "swing peaks after preparation", "swing peak timing is not clear"),
    ]


def _drop_shot(f: Dict[str, float]) -> List[RuleResult]:
    return [
        _rule("high_preparation", _min_score(f["contact_height"], 0.5, 0.7), f["contact_height"], ">= 0.5 torso units", "preparation resembles an overhead stroke", "drop shot preparation should stay high/deceptive"),
        _rule("soft_wrist_speed", _max_score(f["max_wrist_speed"], 0.18, 0.12), f["max_wrist_speed"], "<= 0.18", "wrist speed is controlled", "reduce late wrist speed for a softer drop"),
        _rule("compact_motion", _max_score(f["hip_speed_mean"], 0.05, 0.08), f["hip_speed_mean"], "<= 0.05", "body motion is compact", "excessive body movement may reduce drop-shot control"),
        _rule("arm_extension", _min_score(f["dominant_elbow_max"], 120.0, 40.0), f["dominant_elbow_max"], ">= 120 degrees", "arm shape remains credible", "maintain overhead arm structure before soft contact"),
    ]


def _net_shot(f: Dict[str, float]) -> List[RuleResult]:
    return [
        _rule("low_wrist_speed", _max_score(f["max_wrist_speed"], 0.12, 0.10), f["max_wrist_speed"], "<= 0.12", "touch is soft/controlled", "net shots should use less wrist speed"),
        _rule("compact_body", _max_score(f["hip_speed_mean"], 0.04, 0.08), f["hip_speed_mean"], "<= 0.04", "body movement is compact", "reduce large body movement near the net"),
        _rule("compact_wrist_travel", _max_score(f["wrist_vertical_travel"], 0.8, 0.8), f["wrist_vertical_travel"], "<= 0.8", "wrist travel is compact", "net shot should be a shorter controlled action"),
    ]


def _lift(f: Dict[str, float]) -> List[RuleResult]:
    return [
        _rule("upward_wrist_motion", _min_score(f["wrist_upward_motion"], 0.25, 0.5), f["wrist_upward_motion"], ">= 0.25", "wrist rises into the lift", "emphasize upward lifting action from low contact"),
        _rule("controlled_power", _range_score(f["max_wrist_speed"], 0.04, 0.22, 0.14), f["max_wrist_speed"], "0.04-0.22", "lift speed is controlled", "lift speed is unusually low/high"),
        _rule("body_support", _min_score(f["hip_speed_mean"], 0.005, 0.04), f["hip_speed_mean"], ">= 0.005", "body contributes to lift", "use legs/body support rather than wrist only"),
        _rule("timing", _range_score(f["impact_frame"], 0.20, 0.85, 0.35), f["impact_frame"], "0.20-0.85", "timing is plausible", "timing looks unusual; check clip trimming"),
    ]


_RULES: Dict[str, Callable[[Dict[str, float]], List[RuleResult]]] = {
    "smash": _smash,
    "clear": _clear,
    "drop_shot": _drop_shot,
    "net_shot": _net_shot,
    "lift": _lift,
}


def evaluate_stroke_rules(keypoints: np.ndarray, stroke_type: str) -> RuleEvaluation:
    stroke_type = str(stroke_type)
    if stroke_type not in _RULES:
        raise ValueError(f"Unsupported stroke_type: {stroke_type}. Expected one of {STROKE_TYPES}")
    feats = _features(keypoints)
    results = _RULES[stroke_type](feats)
    overall = float(np.mean([rule.score for rule in results])) if results else 0.0
    feedback = [rule.feedback for rule in results]
    return RuleEvaluation(
        stroke_type=stroke_type,
        overall_score=overall,
        rule_results=results,
        feedback=feedback,
    )


def rule_score_summary(evaluation: RuleEvaluation) -> Dict[str, float]:
    return {result.rule: float(result.score) for result in evaluation.rule_results}
