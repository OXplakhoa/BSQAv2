"""Data Mining chart/table preparation helpers for Streamlit pages.

The Data Mining Motion Lab reuses static analysis artifacts from ``results/`` and
local features from a selected ``PipelineRun``.  These functions intentionally
return plain Python rows so they can be tested without Streamlit.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt


Number = Optional[float]


def _to_float(value: Any) -> Number:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _format_number(value: Number, digits: int = 4) -> str:
    if value is None:
        return "missing"
    return f"{value:.{digits}f}"


def horizontal_bar_figure(
    rows: List[Dict[str, Any]],
    label_key: str,
    value_key: str,
    title: str = "",
    limit: Optional[int] = None,
):
    """Build a small Matplotlib horizontal bar chart for Streamlit display."""
    chart_rows = rows[:limit] if limit is not None else list(rows)
    labels = [str(row.get(label_key, "")) for row in chart_rows]
    values = [_to_float(row.get(value_key)) or 0.0 for row in chart_rows]

    fig_height = max(2.4, 0.36 * max(1, len(chart_rows)))
    fig, ax = plt.subplots(figsize=(7, fig_height))
    positions = list(range(len(labels)))
    ax.barh(positions, values, color="#4C78A8")
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(value_key.replace("_", " "))
    if title:
        ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def load_json_file(path: Path) -> Dict[str, Any]:
    """Load a JSON object, returning ``{}`` when the artifact is absent.

    Missing-result handling belongs here so Streamlit pages can show friendly
    warnings rather than crashing during a live presentation.
    """
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def load_text_file(path: Path) -> str:
    path = Path(path)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def existing_path(path: Path) -> Optional[Path]:
    path = Path(path)
    return path if path.exists() else None


def feature_value_rows(
    features: Dict[str, Any],
    feature_order: Optional[Iterable[str]] = None,
    top_n: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return current-sample feature rows with stable ordering.

    Features present in ``feature_order`` appear first, followed by any extra
    features alphabetically.  This lets the page align with the RF training
    order while still surfacing ad-hoc diagnostic features if present.
    """
    ordered: List[str] = []
    seen = set()
    if feature_order:
        for name in feature_order:
            if name in features and name not in seen:
                ordered.append(name)
                seen.add(name)
    for name in sorted(features):
        if name not in seen:
            ordered.append(name)
            seen.add(name)

    if top_n is not None:
        ordered = ordered[:top_n]

    rows: List[Dict[str, Any]] = []
    for name in ordered:
        value = _to_float(features.get(name))
        rows.append({
            "feature": name,
            "value": value,
            "value_display": _format_number(value),
        })
    return rows


def rf_top_feature_rows(
    rf_results: Dict[str, Any],
    current_features: Dict[str, Any],
    top_n: int = 12,
) -> List[Dict[str, Any]]:
    """Join global RF feature importances with current-sample values."""
    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(rf_results.get("top_features", [])[:top_n], start=1):
        feature = str(item.get("feature", ""))
        current_value = _to_float(current_features.get(feature))
        importance = _to_float(item.get("importance"))
        rows.append({
            "rank": _to_int(item.get("rank"), idx),
            "feature": feature,
            "importance": importance,
            "current_value": current_value,
            "current_value_display": _format_number(current_value),
        })
    return rows


def entropy_class_rows(entropy_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return class-distribution/entropy rows from entropy_analysis.json."""
    distribution = entropy_payload.get("class_distribution", {})
    rows: List[Dict[str, Any]] = []
    for class_name, stats in distribution.items():
        rows.append({
            "class": str(class_name),
            "count": _to_int(stats.get("count")),
            "probability": _to_float(stats.get("prob")) or 0.0,
            "entropy_contribution": _to_float(stats.get("entropy_contribution")) or 0.0,
        })
    return rows


def entropy_information_gain_rows(entropy_payload: Dict[str, Any], top_n: int = 10) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(entropy_payload.get("top_information_gains", [])[:top_n], start=1):
        rows.append({
            "rank": idx,
            "feature": str(item.get("feature", "")),
            "information_gain": _to_float(item.get("information_gain")) or 0.0,
            "normalized_ig": _to_float(item.get("normalized_ig")) or 0.0,
        })
    return rows


def mutual_information_rows(path: Path, top_n: int = 15) -> List[Dict[str, Any]]:
    """Read mutual_information.csv into chart/table rows."""
    path = Path(path)
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for item in reader:
            rows.append({
                "rank": _to_int(item.get("rank"), len(rows) + 1),
                "feature": str(item.get("feature", "")),
                "mutual_information": _to_float(item.get("mutual_information")) or 0.0,
                "normalized": _to_float(item.get("normalized")) or 0.0,
            })
            if len(rows) >= top_n:
                break
    return rows


def class_average_comparison_rows(
    rf_bundle: Any,
    features: Dict[str, Any],
    class_name: str,
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """Return current sample vs RF training class-average comparison rows."""
    comparison = rf_bundle.compare_to_class_average(features, class_name, top_n=top_n)
    rows: List[Dict[str, Any]] = []
    for item in comparison:
        value = _to_float(item.get("value"))
        average = _to_float(item.get("class_average"))
        absolute_delta = _to_float(item.get("absolute_delta"))
        rows.append({
            "feature": str(item.get("feature", "")),
            "value": value,
            "class_average": average,
            "absolute_delta": absolute_delta,
            "value_display": _format_number(value),
            "class_average_display": _format_number(average),
            "absolute_delta_display": _format_number(absolute_delta),
        })
    return rows


def decision_tree_split_rows(decision_tree_results: Dict[str, Any], top_n: int = 10) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(decision_tree_results.get("top_splits", [])[:top_n], start=1):
        rows.append({
            "rank": idx,
            "feature": str(item.get("feature", "")),
            "importance": _to_float(item.get("importance")) or 0.0,
        })
    return rows
