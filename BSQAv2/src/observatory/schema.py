"""Shared data contracts for the Streamlit Pipeline Observatory.

The UI should not depend on ad-hoc dictionaries produced by scripts.  A
PipelineRun is the stable hand-off object between pose extraction,
preprocessing, DL inference, DM inference, diagnostics, and Streamlit pages.
Large numpy arrays are saved as separate .npy files; run.json stores only
metadata and relative array paths.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class ArtifactError(RuntimeError):
    """Base class for artifact layer failures."""


class ArtifactValidationError(ArtifactError):
    """Raised when an artifact exists but does not satisfy the schema."""


@dataclass
class PredictionResult:
    """Model prediction payload used by DL and RF branches."""

    label: Optional[str] = None
    probabilities: Dict[str, float] = field(default_factory=dict)
    confidence: Optional[float] = None
    predicted_index: Optional[int] = None

    @classmethod
    def from_probabilities(
        cls,
        probabilities: Dict[str, float],
        predicted_index: Optional[int] = None,
    ) -> "PredictionResult":
        if not probabilities:
            return cls(predicted_index=predicted_index)
        label, confidence = max(probabilities.items(), key=lambda item: item[1])
        return cls(
            label=label,
            probabilities={str(k): float(v) for k, v in probabilities.items()},
            confidence=float(confidence),
            predicted_index=predicted_index,
        )


@dataclass
class CuratedSample:
    """Manifest entry for one presentation-ready sample."""

    sample_id: str
    title: str
    stroke_type: str
    video_path: str
    ground_truth: Optional[str] = None
    manual_review_status: str = "reviewed"
    teaching_point: str = ""
    diagnosis: str = ""
    pipeline_run_dir: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CuratedSample":
        required = ["sample_id", "title", "stroke_type", "video_path"]
        missing = [name for name in required if not data.get(name)]
        if missing:
            raise ArtifactValidationError(
                f"Curated sample is missing required fields: {', '.join(missing)}"
            )
        return cls(
            sample_id=str(data["sample_id"]),
            title=str(data["title"]),
            stroke_type=str(data["stroke_type"]),
            video_path=str(data["video_path"]),
            ground_truth=data.get("ground_truth"),
            manual_review_status=str(data.get("manual_review_status", "reviewed")),
            teaching_point=str(data.get("teaching_point", "")),
            diagnosis=str(data.get("diagnosis", "")),
            pipeline_run_dir=data.get("pipeline_run_dir"),
            tags=list(data.get("tags", [])),
        )


@dataclass
class PipelineRun:
    """Canonical result object for one video-to-prediction pipeline run."""

    run_id: str
    sample_id: str
    mode: str
    created_at: str
    source_video_path: Optional[str] = None
    ground_truth: Optional[str] = None
    video_metadata: Dict[str, Any] = field(default_factory=dict)
    pose_qc: Dict[str, Any] = field(default_factory=dict)
    dl_prediction: PredictionResult = field(default_factory=PredictionResult)
    rf_prediction: PredictionResult = field(default_factory=PredictionResult)
    dm_features: Dict[str, float] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    timings_ms: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    arrays: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    @classmethod
    def new(
        cls,
        run_id: str,
        sample_id: str,
        mode: str,
        source_video_path: Optional[str] = None,
        ground_truth: Optional[str] = None,
    ) -> "PipelineRun":
        return cls(
            run_id=run_id,
            sample_id=sample_id,
            mode=mode,
            created_at=datetime.now(timezone.utc).isoformat(),
            source_video_path=source_video_path,
            ground_truth=ground_truth,
        )

    def validate(self) -> None:
        """Validate required fields and known array shapes."""
        if not self.run_id:
            raise ArtifactValidationError("PipelineRun.run_id is required")
        if not self.sample_id:
            raise ArtifactValidationError("PipelineRun.sample_id is required")
        if not self.mode:
            raise ArtifactValidationError("PipelineRun.mode is required")

        normalized = self.arrays.get("normalized_keypoints")
        if normalized is not None and normalized.shape != (64, 17, 2):
            raise ArtifactValidationError(
                "normalized_keypoints must have shape (64, 17, 2); "
                f"got {normalized.shape}"
            )

        raw = self.arrays.get("raw_keypoints")
        if raw is not None and (raw.ndim != 3 or raw.shape[1:] != (17, 2)):
            raise ArtifactValidationError(
                "raw_keypoints must have shape (T, 17, 2); "
                f"got {raw.shape}"
            )


def _json_safe(value: Any) -> Any:
    """Convert numpy/scalar/path values to JSON-safe native values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def save_pipeline_run(run: PipelineRun, run_dir: Path) -> Path:
    """Save PipelineRun metadata and arrays under run_dir.

    Returns the path to run.json.
    """
    run.validate()
    run_dir = Path(run_dir)
    arrays_dir = run_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)

    array_files: Dict[str, str] = {}
    for key, value in sorted(run.arrays.items()):
        if value is None:
            continue
        arr = np.asarray(value)
        array_path = arrays_dir / f"{key}.npy"
        np.save(array_path, arr)
        array_files[key] = array_path.relative_to(run_dir).as_posix()

    metadata = asdict(run)
    metadata.pop("arrays", None)
    metadata["array_files"] = array_files

    json_path = run_dir / "run.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(metadata), f, indent=2)
    return json_path


def load_pipeline_run(run_dir: Path, load_arrays: bool = True) -> PipelineRun:
    """Load a PipelineRun from a directory containing run.json."""
    run_dir = Path(run_dir)
    json_path = run_dir / "run.json"
    if not json_path.exists():
        raise ArtifactValidationError(f"Missing PipelineRun metadata: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    array_files = metadata.pop("array_files", {})
    dl_pred = PredictionResult(**metadata.pop("dl_prediction", {}))
    rf_pred = PredictionResult(**metadata.pop("rf_prediction", {}))

    run = PipelineRun(
        **metadata,
        dl_prediction=dl_pred,
        rf_prediction=rf_pred,
        arrays={},
    )

    if load_arrays:
        for key, relative_path in array_files.items():
            array_path = run_dir / relative_path
            if not array_path.exists():
                raise ArtifactValidationError(f"Missing array file: {array_path}")
            run.arrays[key] = np.load(array_path, allow_pickle=False)

    run.validate()
    return run
