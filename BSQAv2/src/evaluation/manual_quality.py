"""Manual quality-label subset evaluation helpers."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _float(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _rankdata(values: List[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr)
    ranks = np.empty(len(arr), dtype=float)
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = ((i + j) / 2.0) + 1.0
        i = j + 1
    return ranks


def spearman_rs(x: List[float], y: List[float]) -> float | None:
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return None
    rx = _rankdata(x)
    ry = _rankdata(y)
    if float(np.std(rx)) == 0.0 or float(np.std(ry)) == 0.0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def load_manual_quality_labels(path: Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            score = _float(row.get("quality_score_0_100"))
            if score is None:
                continue
            rows.append({**row, "quality_score_0_100": score})
    return rows


def load_manual_quality_results(path: Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            manual = _float(row.get("manual_quality"))
            predicted = _float(row.get("predicted_quality"))
            if manual is None or predicted is None:
                continue
            abs_error = _float(row.get("absolute_error"), abs(predicted - manual))
            rows.append({
                **row,
                "manual_quality": manual,
                "predicted_quality": predicted,
                "absolute_error": abs_error,
            })
    return rows


def summarize_manual_quality_results(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"summary": {"n_samples": 0}, "per_stroke": []}

    manual = [float(row["manual_quality"]) for row in rows]
    predicted = [float(row["predicted_quality"]) for row in rows]
    errors = [float(row["absolute_error"]) for row in rows]

    per_stroke: List[Dict[str, Any]] = []
    for stroke in sorted({str(row.get("stroke_type")) for row in rows}):
        subset = [row for row in rows if row.get("stroke_type") == stroke]
        stroke_manual = [float(row["manual_quality"]) for row in subset]
        stroke_pred = [float(row["predicted_quality"]) for row in subset]
        stroke_errors = [float(row["absolute_error"]) for row in subset]
        per_stroke.append({
            "stroke_type": stroke,
            "n_samples": len(subset),
            "manual_mean": float(np.mean(stroke_manual)),
            "predicted_mean": float(np.mean(stroke_pred)),
            "mae": float(np.mean(stroke_errors)),
            "spearman_rs": spearman_rs(stroke_manual, stroke_pred),
        })

    return {
        "summary": {
            "n_samples": len(rows),
            "manual_mean": float(np.mean(manual)),
            "predicted_mean": float(np.mean(predicted)),
            "mae": float(np.mean(errors)),
            "spearman_rs": spearman_rs(manual, predicted),
            "label_source": "manual 0-100 quality labels",
            "prediction_source": "Phase 4 heuristic quality scorer over extracted MediaPipe skeletons",
        },
        "per_stroke": per_stroke,
    }
