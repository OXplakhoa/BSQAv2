"""Chart data preparation for Streamlit pages.

These helpers return plain dictionaries/lists so they can be unit-tested without
Streamlit.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .bootstrap import ensure_project_imports

ensure_project_imports()

from src.data.skeleton import KEYPOINT_NAMES
from src.observatory.schema import PipelineRun


CRITICAL_JOINTS = [5, 6, 7, 8, 9, 10]


def probability_rows(probabilities: Dict[str, float]) -> List[Dict[str, float]]:
    rows = [
        {"class": label, "probability": float(value)}
        for label, value in probabilities.items()
    ]
    rows.sort(key=lambda row: row["probability"], reverse=True)
    return rows


def prediction_summary_rows(run: PipelineRun) -> List[Dict[str, object]]:
    return [
        {
            "branch": "Random Forest",
            "prediction": run.rf_prediction.label,
            "confidence": run.rf_prediction.confidence,
        },
        {
            "branch": "Deep Learning",
            "prediction": run.dl_prediction.label,
            "confidence": run.dl_prediction.confidence,
        },
    ]


def missing_joint_ratio_rows(run: PipelineRun) -> List[Dict[str, float]]:
    ratios = run.pose_qc.get("missing_joint_ratio_by_frame", [])
    return [
        {"frame": idx, "missing_joint_ratio": float(value)}
        for idx, value in enumerate(ratios)
    ]


def critical_visibility_rows(run: PipelineRun) -> List[Dict[str, float]]:
    visibility = run.pose_qc.get("critical_joint_visibility", {})
    return [
        {"joint": joint, "visibility": float(value)}
        for joint, value in visibility.items()
    ]


def skeleton_frame_rows(run: PipelineRun, frame_index: int, normalized: bool = False) -> List[Dict[str, float]]:
    key = "normalized_keypoints" if normalized else "raw_keypoints"
    keypoints = run.arrays.get(key)
    if keypoints is None:
        return []
    frame_index = int(np.clip(frame_index, 0, keypoints.shape[0] - 1))
    frame = keypoints[frame_index]
    return [
        {
            "joint": KEYPOINT_NAMES[idx],
            "joint_index": idx,
            "x": float(x),
            "y": float(y),
            "critical": idx in CRITICAL_JOINTS,
        }
        for idx, (x, y) in enumerate(frame)
    ]


def attention_frame_importance(run: PipelineRun) -> List[Dict[str, float]]:
    attention = run.arrays.get("attention_weights")
    if attention is None or attention.size == 0:
        return []
    # Mean incoming attention per frame gives a simple frame-importance curve.
    importance = np.asarray(attention).mean(axis=0)
    return [
        {"frame": idx, "attention": float(value)}
        for idx, value in enumerate(importance)
    ]
