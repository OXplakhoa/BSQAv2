"""Deep Learning visualization helpers for Streamlit pages."""
from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from .bootstrap import ensure_project_imports

ensure_project_imports()

from src.observatory.schema import PipelineRun, PredictionResult


def top_attention_frames(run: PipelineRun, top_n: int = 8) -> List[Dict[str, float]]:
    """Return top frames by mean incoming temporal attention."""
    attention = run.arrays.get("attention_weights")
    if attention is None or attention.size == 0:
        return []
    importance = np.asarray(attention, dtype=np.float32).mean(axis=0)
    indices = np.argsort(importance)[::-1][:top_n]
    return [
        {"rank": rank + 1, "frame": int(idx), "attention": float(importance[idx])}
        for rank, idx in enumerate(indices)
    ]


def confidence_interpretation(prediction: PredictionResult) -> str:
    """Plain-language confidence interpretation for the DL branch."""
    confidence = prediction.confidence
    if confidence is None:
        return "DL confidence is unavailable for this sample."
    if confidence >= 0.70:
        return "High DL confidence: the model has a clear top class, but pose quality should still be checked."
    if confidence >= 0.50:
        return "Medium DL confidence: the model has a preferred class, but the case may be ambiguous."
    return "Low DL confidence: treat the DL prediction cautiously and compare it with RF and pose quality."


def attention_heatmap_figure(attention: np.ndarray):
    """Create a Matplotlib heatmap for temporal attention weights."""
    attention = np.asarray(attention, dtype=np.float32)
    if attention.ndim != 2:
        raise ValueError(f"attention must be 2D; got shape {attention.shape}")

    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(attention, aspect="auto", cmap="viridis")
    ax.set_title("Temporal attention heatmap")
    ax.set_xlabel("Key frame attended to")
    ax.set_ylabel("Query frame")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="attention")
    fig.tight_layout()
    return fig


def dl_shape_rows(run: PipelineRun) -> List[Dict[str, str]]:
    shapes = run.diagnostics.get("dl_shapes", {})
    return [
        {"tensor": name, "shape": str(value)}
        for name, value in shapes.items()
    ]
