"""Custom upload pipeline helpers for the Streamlit Observatory.

This module keeps MediaPipe/live-video handling behind a small testable
interface. Tests inject a fake extractor; the Streamlit page uses the default
MediaPipe extractor only after a user explicitly uploads a video.
"""
from __future__ import annotations

from pathlib import Path
import re
from typing import Callable, Dict, Iterable, Optional, Tuple, Union
from uuid import uuid4

import numpy as np

from .pipeline import run_skeleton_pipeline
from .schema import PipelineRun

QualityReferences = Union[Iterable[Tuple[str, np.ndarray]], Dict[str, Iterable[Tuple[str, np.ndarray]]]]

Extractor = Callable[[Path], Tuple[np.ndarray, np.ndarray, int]]


def _safe_filename(filename: str) -> str:
    name = Path(filename or "upload.mp4").name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return name or "upload.mp4"


def save_uploaded_file(data: bytes, upload_dir: Path, filename: str) -> Path:
    """Persist uploaded bytes under upload_dir with a safe unique filename."""
    upload_dir = Path(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_filename(filename)
    path = upload_dir / f"{uuid4().hex[:8]}_{safe_name}"
    path.write_bytes(data)
    return path


def _default_extractor(video_path: Path) -> Tuple[np.ndarray, np.ndarray, int]:
    from src.utils.video_to_csv import extract_keypoints_from_video

    keypoints, visibilities, fps = extract_keypoints_from_video(str(video_path))
    return keypoints, visibilities, fps


def run_uploaded_video_pipeline(
    video_path: Path,
    sample_id: str,
    ground_truth: Optional[str] = None,
    rf_bundle_path: Optional[Path] = None,
    dl_checkpoint_path: Optional[Path] = None,
    dl_device: str = "cpu",
    extractor: Optional[Extractor] = None,
    quality_references: Optional[QualityReferences] = None,
    run_id: Optional[str] = None,
) -> PipelineRun:
    """Extract pose from an uploaded video and return a PipelineRun.

    The returned object uses ``mode='upload'`` and the same PipelineRun contract
    as curated cached samples.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Uploaded video not found: {video_path}")

    extractor = extractor or _default_extractor
    keypoints, visibilities, fps = extractor(video_path)

    run = run_skeleton_pipeline(
        sample_id=sample_id,
        raw_keypoints=keypoints,
        visibilities=visibilities,
        source_video_path=str(video_path),
        ground_truth=ground_truth,
        rf_bundle_path=rf_bundle_path,
        dl_checkpoint_path=dl_checkpoint_path,
        dl_device=dl_device,
        quality_references=quality_references,
        run_id=run_id or f"{sample_id}_{uuid4().hex[:8]}",
    )
    run.mode = "upload"
    run.video_metadata["fps"] = int(fps)
    run.video_metadata["uploaded_video_path"] = str(video_path)
    run.validate()
    return run
