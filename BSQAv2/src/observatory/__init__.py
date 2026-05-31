"""Pipeline Observatory package.

Modules here support the Streamlit demo artifact layer and inference contracts.
"""

from .artifacts import ArtifactRegistry, bootstrap_artifact_layout
from .dl_inference import DLInferenceResult, DLModelBundle, load_dl_model, run_dl_inference
from .pipeline import run_skeleton_pipeline
from .pose_quality import evaluate_pose_quality
from .schema import (
    ArtifactError,
    ArtifactValidationError,
    CuratedSample,
    PipelineRun,
    PredictionResult,
    load_pipeline_run,
    save_pipeline_run,
)

__all__ = [
    "ArtifactRegistry",
    "ArtifactError",
    "ArtifactValidationError",
    "CuratedSample",
    "DLInferenceResult",
    "DLModelBundle",
    "PipelineRun",
    "PredictionResult",
    "bootstrap_artifact_layout",
    "evaluate_pose_quality",
    "load_dl_model",
    "load_pipeline_run",
    "run_dl_inference",
    "run_skeleton_pipeline",
    "save_pipeline_run",
]
