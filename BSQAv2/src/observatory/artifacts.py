"""Artifact registry for the BSQAv2 Pipeline Observatory.

This module centralizes paths for curated samples, cached PipelineRun objects,
trained model artifacts, metrics, and pre-rendered figures. Streamlit pages use
this registry instead of hard-coding storage layout.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from src.config import PROJECT_ROOT
from .schema import (
    ArtifactValidationError,
    CuratedSample,
    PipelineRun,
    load_pipeline_run,
    save_pipeline_run,
)


ARTIFACT_ROOT = PROJECT_ROOT / "webapp" / "artifacts"
CURATED_DIR = ARTIFACT_ROOT / "curated"
PIPELINE_RUNS_DIR = ARTIFACT_ROOT / "pipeline_runs"
MODELS_DIR = ARTIFACT_ROOT / "models"
METRICS_DIR = ARTIFACT_ROOT / "metrics"
FIGURES_DIR = ARTIFACT_ROOT / "figures"
FEATURE_SPACE_DIR = ARTIFACT_ROOT / "feature_space"
CURATED_MANIFEST = CURATED_DIR / "manifest.json"


class ArtifactRegistry:
    """Discover, validate, load, and save observatory artifacts."""

    def __init__(self, root: Path = ARTIFACT_ROOT):
        self.root = Path(root)
        self.curated_dir = self.root / "curated"
        self.pipeline_runs_dir = self.root / "pipeline_runs"
        self.models_dir = self.root / "models"
        self.metrics_dir = self.root / "metrics"
        self.figures_dir = self.root / "figures"
        self.feature_space_dir = self.root / "feature_space"
        self.curated_manifest = self.curated_dir / "manifest.json"

    def ensure_layout(self) -> None:
        """Create the artifact directory skeleton if it does not exist."""
        for path in [
            self.curated_dir,
            self.pipeline_runs_dir,
            self.models_dir,
            self.metrics_dir,
            self.figures_dir,
            self.feature_space_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def list_curated_samples(self) -> List[CuratedSample]:
        """Load and validate curated sample manifest entries."""
        if not self.curated_manifest.exists():
            return []
        with self.curated_manifest.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, dict):
            entries = payload.get("samples", [])
        elif isinstance(payload, list):
            entries = payload
        else:
            raise ArtifactValidationError(
                f"Curated manifest must be a dict or list: {self.curated_manifest}"
            )

        return [CuratedSample.from_dict(entry) for entry in entries]

    def get_curated_sample(self, sample_id: str) -> CuratedSample:
        for sample in self.list_curated_samples():
            if sample.sample_id == sample_id:
                return sample
        raise ArtifactValidationError(f"Unknown curated sample_id: {sample_id}")

    def pipeline_run_dir(self, run_id: str) -> Path:
        return self.pipeline_runs_dir / run_id

    def save_pipeline_run(self, run: PipelineRun) -> Path:
        self.ensure_layout()
        return save_pipeline_run(run, self.pipeline_run_dir(run.run_id))

    def load_pipeline_run(self, run_id: str, load_arrays: bool = True) -> PipelineRun:
        return load_pipeline_run(self.pipeline_run_dir(run_id), load_arrays=load_arrays)

    def resolve_rf_bundle(self) -> Path:
        """Return default RF model bundle path, with helpful validation."""
        path = self.models_dir / "rf_baseline" / "rf_model_bundle.joblib"
        if not path.exists():
            raise ArtifactValidationError(
                "Random Forest bundle not found. Generate it with: "
                "python src/data/rf_baseline.py --export-artifact"
            )
        return path

    def resolve_metrics(self, name: str) -> Optional[Path]:
        path = self.metrics_dir / name
        return path if path.exists() else None

    def resolve_figure(self, name: str) -> Optional[Path]:
        path = self.figures_dir / name
        return path if path.exists() else None


def bootstrap_artifact_layout(root: Path = ARTIFACT_ROOT) -> ArtifactRegistry:
    """Create artifact directories and return a registry instance."""
    registry = ArtifactRegistry(root)
    registry.ensure_layout()
    return registry
