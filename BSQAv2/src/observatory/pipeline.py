"""Canonical skeleton-to-artifact pipeline for the Observatory.

This module is intentionally video-agnostic for the first foundation slice. It
accepts raw COCO-17 skeleton arrays, computes pose quality, preprocesses to the
64-frame model tensor, runs the Data Mining/RF branch, and returns a PipelineRun.
A later MediaPipe adapter can feed this same interface from uploaded videos.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union
from uuid import uuid4

import numpy as np

from src.config import SEQUENCE_LENGTH
from src.data.biomechanics import extract_features
from src.data.preprocessing import preprocess_sequence

from .diagnostics import compare_branches, summarize_prediction
from .dl_inference import load_dl_model, run_dl_inference
from .dm_inference import load_rf_bundle
from .pose_quality import evaluate_pose_quality
from .schema import PipelineRun
from src.quality.hybrid import HybridQualityScorer


def run_skeleton_pipeline(
    sample_id: str,
    raw_keypoints: np.ndarray,
    visibilities: Optional[np.ndarray] = None,
    source_video_path: Optional[str] = None,
    ground_truth: Optional[str] = None,
    rf_bundle_path: Optional[Path] = None,
    dl_checkpoint_path: Optional[Path] = None,
    dl_device: str = "cpu",
    quality_references: Optional[Union[Iterable[Tuple[str, np.ndarray]], Dict[str, Iterable[Tuple[str, np.ndarray]]]]] = None,
    run_id: Optional[str] = None,
) -> PipelineRun:
    """Run the canonical pipeline from skeleton arrays to PipelineRun.

    Args:
        sample_id: Stable identifier for the selected clip/sample.
        raw_keypoints: Raw COCO-17 keypoints shaped (T, 17, 2), pixel space.
        visibilities: Optional visibility array shaped (T, 17).
        source_video_path: Optional source video path for provenance.
        ground_truth: Optional true stroke label. If absent, diagnostics must not
            claim correctness.
        rf_bundle_path: Optional path to the exported RF artifact. When present,
            the Data Mining branch runs RF inference.
        dl_checkpoint_path: Optional path to GCN+BiLSTM+Attention checkpoint.
            When present, the Deep Learning branch runs inference and stores
            temporal attention weights.
        dl_device: Torch device for DL inference, normally "cpu" for demo setup.
        quality_references: Optional same-stroke reference skeletons for DTW
            quality scoring, or a dict mapping stroke_type -> references. If
            omitted, a rule-only quality report is produced.
        run_id: Optional deterministic run ID for cached artifacts/tests.

    Returns:
        PipelineRun with raw and normalized arrays stored separately.
    """
    raw_keypoints = np.asarray(raw_keypoints, dtype=np.float32)
    if raw_keypoints.ndim != 3 or raw_keypoints.shape[1:] != (17, 2):
        raise ValueError(
            f"raw_keypoints must have shape (T, 17, 2); got {raw_keypoints.shape}"
        )

    run = PipelineRun.new(
        run_id=run_id or f"{sample_id}_{uuid4().hex[:8]}",
        sample_id=sample_id,
        mode="skeleton",
        source_video_path=source_video_path,
        ground_truth=ground_truth,
    )

    if source_video_path is not None:
        run.video_metadata["source_video_path"] = source_video_path
    run.video_metadata["raw_frame_count"] = int(raw_keypoints.shape[0])

    run.pose_qc = evaluate_pose_quality(raw_keypoints, visibilities=visibilities)
    normalized_keypoints = preprocess_sequence(raw_keypoints, target_length=SEQUENCE_LENGTH)

    run.arrays["raw_keypoints"] = raw_keypoints
    run.arrays["normalized_keypoints"] = normalized_keypoints.astype(np.float32)
    if visibilities is not None:
        run.arrays["visibilities"] = np.asarray(visibilities, dtype=np.float32)

    run.dm_features = extract_features(normalized_keypoints)

    if rf_bundle_path is not None:
        rf_bundle = load_rf_bundle(Path(rf_bundle_path))
        run.rf_prediction = rf_bundle.predict(run.dm_features)
        run.diagnostics["rf_summary"] = summarize_prediction(
            run.rf_prediction,
            ground_truth=ground_truth,
            pose_reliability=run.pose_qc.get("reliability_score"),
            branch_name="Random Forest",
        )

    if dl_checkpoint_path is not None:
        dl_bundle = load_dl_model(Path(dl_checkpoint_path), device=dl_device)
        dl_result = run_dl_inference(dl_bundle, run.arrays["normalized_keypoints"])
        run.dl_prediction = dl_result.prediction
        run.arrays["attention_weights"] = dl_result.attention_weights
        run.diagnostics["dl_summary"] = summarize_prediction(
            run.dl_prediction,
            ground_truth=ground_truth,
            pose_reliability=run.pose_qc.get("reliability_score"),
            branch_name="Deep Learning",
        )
        run.diagnostics["dl_shapes"] = dl_result.shape_metadata
        if dl_result.quality_score is not None:
            run.diagnostics["dl_quality_score"] = dl_result.quality_score

    run.diagnostics["pose_summary"] = (
        f"Pose reliability is {run.pose_qc.get('reliability_label', 'unknown')} "
        f"({run.pose_qc.get('reliability_score', 0.0):.2f})."
    )
    if run.dl_prediction.label or run.rf_prediction.label:
        run.diagnostics["branch_comparison"] = compare_branches(
            run.dl_prediction,
            run.rf_prediction,
        )

    quality_stroke = ground_truth or run.rf_prediction.label or run.dl_prediction.label
    if quality_stroke:
        references_for_stroke = quality_references or []
        if isinstance(quality_references, dict):
            references_for_stroke = quality_references.get(quality_stroke, [])
        quality_report = HybridQualityScorer().score(
            run.arrays["normalized_keypoints"],
            quality_stroke,
            references=references_for_stroke,
        )
        run.diagnostics["quality_report"] = quality_report
        run.diagnostics["quality_summary"] = (
            f"Technique quality estimate for {quality_stroke}: "
            f"{quality_report['quality_score']:.0f}/100 "
            f"(rules={quality_report['rule_score']:.0f}"
            + (
                f", dtw={quality_report['dtw_score']:.0f})."
                if quality_report.get("dtw_score") is not None
                else ", DTW reference unavailable)."
            )
        )

    run.validate()
    return run
