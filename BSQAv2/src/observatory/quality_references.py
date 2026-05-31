"""Load DTW reference skeletons from curated PipelineRun artifacts."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Tuple

import numpy as np

from src.config import PROJECT_ROOT, STROKE_TYPES

from .artifacts import ArtifactRegistry
from .schema import CuratedSample, PipelineRun, load_pipeline_run

QualityReferenceBank = Dict[str, List[Tuple[str, np.ndarray]]]


def _resolve_pipeline_run_dir(registry: ArtifactRegistry, sample: CuratedSample) -> Optional[Path]:
    if not sample.pipeline_run_dir:
        return None

    raw = Path(sample.pipeline_run_dir)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([
            PROJECT_ROOT / raw,
            registry.root / raw,
            registry.pipeline_runs_dir / raw.name,
        ])

    for path in candidates:
        if (path / "run.json").exists():
            return path
    return None


def _reference_array(run: PipelineRun) -> Optional[np.ndarray]:
    arr = run.arrays.get("normalized_keypoints")
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape != (64, 17, 2):
        return None
    return arr


def load_quality_reference_bank(
    registry: Optional[ArtifactRegistry] = None,
    max_per_stroke: int = 3,
    min_pose_reliability: float = 0.0,
) -> QualityReferenceBank:
    """Build a per-stroke DTW reference bank from curated cached runs.

    The bank is intentionally small by default so live upload scoring remains
    fast. Each value is a list of ``(reference_id, normalized_keypoints)`` pairs
    suitable for ``HybridQualityScorer.score(..., references=...)``.
    """
    registry = registry or ArtifactRegistry()
    max_per_stroke = max(1, int(max_per_stroke))
    bank: DefaultDict[str, List[Tuple[str, np.ndarray]]] = defaultdict(list)

    for sample in registry.list_curated_samples():
        stroke = sample.ground_truth or sample.stroke_type
        if stroke not in STROKE_TYPES:
            continue
        if len(bank.get(stroke, [])) >= max_per_stroke:
            continue
        run_dir = _resolve_pipeline_run_dir(registry, sample)
        if run_dir is None:
            continue
        try:
            run = load_pipeline_run(run_dir, load_arrays=True)
        except Exception:
            continue
        reliability = float(run.pose_qc.get("reliability_score", 0.0) or 0.0)
        if reliability < min_pose_reliability:
            continue
        arr = _reference_array(run)
        if arr is None:
            continue
        bank[stroke].append((sample.sample_id, arr))

    return {stroke: refs for stroke, refs in bank.items()}


def reference_bank_summary(bank: QualityReferenceBank) -> Dict[str, int]:
    """Return a compact stroke -> reference count summary."""
    return {stroke: len(bank.get(stroke, [])) for stroke in STROKE_TYPES}
