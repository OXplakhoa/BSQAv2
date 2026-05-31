"""Training/evaluation dashboard data helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


STROKE_ORDER = ["smash", "clear", "drop_shot", "net_shot", "lift"]


# Defense/report values from docs/STREAMLIT_PIPELINE_OBSERVATORY_PRD.md.
# The local demo checkpoint folder may contain a weaker exploratory run; these
# values are the final selected DL v5 metrics for presentation.
FINAL_DL_METRICS = {
    "model": "GCN + BiLSTM + Attention",
    "accuracy": 0.6563,
    "accuracy_std": 0.0372,
    "f1_macro": 0.6483,
    "f1_macro_std": 0.0408,
    "f1_weighted": 0.6517,
    "f1_weighted_std": 0.0400,
    "per_class_f1": {
        "smash": 0.824,
        "clear": 0.644,
        "drop_shot": 0.690,
        "net_shot": 0.758,
        "lift": 0.434,
    },
}


def _float(payload: Dict[str, Any], key: str, default: Optional[float] = None) -> Optional[float]:
    value = payload.get(key, default)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def rf_summary_metrics(rf_results: Dict[str, Any]) -> Dict[str, Optional[float]]:
    return {
        "accuracy": _float(rf_results, "accuracy"),
        "f1_macro": _float(rf_results, "f1_macro"),
        "f1_weighted": _float(rf_results, "f1_weighted"),
    }


def dl_final_metrics() -> Dict[str, Any]:
    return {
        **FINAL_DL_METRICS,
        "per_class_f1": dict(FINAL_DL_METRICS["per_class_f1"]),
    }


def comparison_rows(
    rf_results: Dict[str, Any],
    decision_tree_results: Dict[str, Any],
    dl_metrics: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    dl_metrics = dl_metrics or dl_final_metrics()
    rows: List[Dict[str, Any]] = []

    if dl_metrics:
        rows.append({
            "model": "GCN + BiLSTM + Attention",
            "family": "Deep Learning",
            "accuracy": _float(dl_metrics, "accuracy"),
            "accuracy_std": _float(dl_metrics, "accuracy_std"),
            "f1_macro": _float(dl_metrics, "f1_macro"),
            "f1_weighted": _float(dl_metrics, "f1_weighted"),
            "note": "Final spatial-temporal architecture; attention supports inspection.",
        })

    if rf_results:
        rows.append({
            "model": "Random Forest",
            "family": "Data Mining",
            "accuracy": _float(rf_results, "accuracy"),
            "accuracy_std": None,
            "f1_macro": _float(rf_results, "f1_macro"),
            "f1_weighted": _float(rf_results, "f1_weighted"),
            "note": "Strongest numerical classifier in current artifacts.",
        })

    if decision_tree_results:
        rows.append({
            "model": "Decision Tree",
            "family": "Data Mining",
            "accuracy": _float(decision_tree_results, "cv_accuracy_mean"),
            "accuracy_std": _float(decision_tree_results, "cv_accuracy_std"),
            "f1_macro": None,
            "f1_weighted": None,
            "note": "Interpretable baseline/rules, not strongest classifier.",
        })

    return rows


def per_class_metric_rows(rf_results: Dict[str, Any], dl_metrics: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    dl_metrics = dl_metrics or dl_final_metrics()
    rf_per_class = rf_results.get("per_class", {}) if rf_results else {}
    dl_per_class = dl_metrics.get("per_class_f1", {}) if dl_metrics else {}
    classes = [cls for cls in STROKE_ORDER if cls in rf_per_class or cls in dl_per_class]
    for cls in sorted(set(rf_per_class) | set(dl_per_class)):
        if cls not in classes:
            classes.append(cls)

    rows: List[Dict[str, Any]] = []
    for cls in classes:
        rf_metrics = rf_per_class.get(cls, {})
        rows.append({
            "class": cls,
            "rf_f1": _float(rf_metrics, "f1-score"),
            "rf_precision": _float(rf_metrics, "precision"),
            "rf_recall": _float(rf_metrics, "recall"),
            "rf_support": _float(rf_metrics, "support"),
            "dl_f1": _float(dl_per_class, cls),
        })
    return rows


def fold_artifact_rows(cv_summary: Dict[str, Any], checkpoint_names: Iterable[str]) -> List[Dict[str, Any]]:
    names = sorted([Path(name).name for name in checkpoint_names])
    n_folds = int(cv_summary.get("n_folds") or len(names) or 0)
    rows: List[Dict[str, Any]] = []
    for fold_idx in range(n_folds):
        checkpoint = next((name for name in names if f"fold{fold_idx}" in name), "missing")
        rows.append({
            "fold": fold_idx,
            "checkpoint": checkpoint,
            "cv_accuracy_mean": _float(cv_summary, "accuracy_mean"),
            "cv_accuracy_std": _float(cv_summary, "accuracy_std"),
            "cv_f1_macro_mean": _float(cv_summary, "f1_macro_mean"),
            "cv_f1_macro_std": _float(cv_summary, "f1_macro_std"),
        })
    return rows


def metric_card_value(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "missing"
    return f"{value:.{digits}f}"
