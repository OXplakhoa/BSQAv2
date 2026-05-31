"""Data-loading helpers shared by Streamlit pages."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from .bootstrap import ensure_project_imports

ensure_project_imports()

from src.observatory.artifacts import ArtifactRegistry
from src.observatory.schema import CuratedSample, PipelineRun, load_pipeline_run


@dataclass
class SelectedCase:
    sample: CuratedSample
    run: PipelineRun


def get_registry() -> ArtifactRegistry:
    return ArtifactRegistry()


def load_curated_samples(registry: Optional[ArtifactRegistry] = None) -> List[CuratedSample]:
    registry = registry or get_registry()
    return registry.list_curated_samples()


def sample_label(sample: CuratedSample) -> str:
    title = sample.title or sample.sample_id
    return f"{title} - {sample.sample_id}"


def resolve_pipeline_run_dir(sample: CuratedSample, registry: Optional[ArtifactRegistry] = None) -> Path:
    registry = registry or get_registry()
    if not sample.pipeline_run_dir:
        raise FileNotFoundError(f"Curated sample has no pipeline_run_dir: {sample.sample_id}")
    path = Path(sample.pipeline_run_dir)
    if not path.is_absolute():
        path = registry.root.parent.parent / path
    if not path.exists():
        raise FileNotFoundError(f"PipelineRun directory not found: {path}")
    return path


def load_selected_case(sample_id: str, registry: Optional[ArtifactRegistry] = None) -> SelectedCase:
    registry = registry or get_registry()
    sample = registry.get_curated_sample(sample_id)
    run_dir = resolve_pipeline_run_dir(sample, registry)
    return SelectedCase(sample=sample, run=load_pipeline_run(run_dir))


def default_sample_id(samples: List[CuratedSample]) -> Optional[str]:
    return samples[0].sample_id if samples else None
