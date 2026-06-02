"""Training/evaluation dashboard data helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt

from src.evaluation.ablation_metrics import ABLATION_MODEL_ROWS, QUALITY_METRIC_NOTE


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
    "quality_mae": "N/A",
    "quality_spearman_rs": "N/A",
    "inference_ms_per_frame": 0.0647,
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
            "quality_mae": dl_metrics.get("quality_mae", "N/A"),
            "quality_spearman_rs": dl_metrics.get("quality_spearman_rs", "N/A"),
            "inference_ms_per_frame": _float(dl_metrics, "inference_ms_per_frame"),
            "note": "Final spatial-temporal architecture; attention supports inspection; quality head not expert-supervised.",
        })

    rows.extend(dict(row) for row in ABLATION_MODEL_ROWS)

    if rf_results:
        rows.append({
            "model": "Random Forest",
            "family": "Data Mining",
            "accuracy": _float(rf_results, "accuracy"),
            "accuracy_std": None,
            "f1_macro": _float(rf_results, "f1_macro"),
            "f1_weighted": _float(rf_results, "f1_weighted"),
            "quality_mae": "N/A",
            "quality_spearman_rs": "N/A",
            "inference_ms_per_frame": None,
            "note": "Strongest non-DL baseline in current artifacts.",
        })

    if decision_tree_results:
        rows.append({
            "model": "Decision Tree",
            "family": "Data Mining",
            "accuracy": _float(decision_tree_results, "cv_accuracy_mean"),
            "accuracy_std": _float(decision_tree_results, "cv_accuracy_std"),
            "f1_macro": None,
            "f1_weighted": None,
            "quality_mae": "N/A",
            "quality_spearman_rs": "N/A",
            "inference_ms_per_frame": None,
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


TRAINING_SCALAR_TAGS = [
    "Loss/train",
    "Loss/val",
    "Acc/train",
    "Acc/val",
    "F1/val_macro",
    "LR",
]


def discover_training_run_dirs(*roots: Path) -> List[Path]:
    """Return run directories that contain fold TensorBoard event files."""
    run_dirs: List[Path] = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        if list(root.glob("fold_*/events.out.tfevents.*")):
            run_dirs.append(root)
        for child in sorted(root.iterdir()):
            if child.is_dir() and list(child.glob("fold_*/events.out.tfevents.*")):
                run_dirs.append(child)
    # Deduplicate while preserving sorted/newest-ish order by path name.
    unique: List[Path] = []
    seen = set()
    for path in sorted(run_dirs, key=lambda p: p.name, reverse=True):
        resolved = str(path.resolve())
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def _fold_from_event_path(event_path: Path) -> str:
    parent = event_path.parent.name
    if parent.startswith("fold_"):
        return parent.replace("fold_", "")
    return parent


def load_tensorboard_scalars(run_dir: Path, tags: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Load scalar curves from TensorBoard event files under a training run.

    Returns an empty list when TensorBoard is unavailable, event files are
    missing, or the event files contain no scalar tags.
    """
    run_dir = Path(run_dir)
    tags_filter = set(tags or TRAINING_SCALAR_TAGS)
    rows: List[Dict[str, Any]] = []
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception:
        return rows

    for event_path in sorted(run_dir.glob("fold_*/events.out.tfevents.*")):
        try:
            accumulator = event_accumulator.EventAccumulator(
                str(event_path),
                size_guidance={"scalars": 0},
            )
            accumulator.Reload()
        except Exception:
            continue

        scalar_tags = accumulator.Tags().get("scalars", [])
        for tag in scalar_tags:
            if tag not in tags_filter:
                continue
            try:
                events = accumulator.Scalars(tag)
            except Exception:
                continue
            for event in events:
                rows.append({
                    "run": run_dir.name,
                    "fold": _fold_from_event_path(event_path),
                    "tag": tag,
                    "step": int(event.step),
                    "value": float(event.value),
                    "wall_time": float(event.wall_time),
                    "event_file": event_path.name,
                })
    return rows


def available_training_curve_tags(rows: Iterable[Dict[str, Any]]) -> List[str]:
    order = {tag: idx for idx, tag in enumerate(TRAINING_SCALAR_TAGS)}
    tags = sorted({str(row.get("tag")) for row in rows if row.get("tag")}, key=lambda t: order.get(t, 999))
    return tags


def available_training_curve_folds(rows: Iterable[Dict[str, Any]]) -> List[str]:
    def _fold_key(value: str):
        try:
            return int(value)
        except ValueError:
            return value
    return sorted({str(row.get("fold")) for row in rows if row.get("fold") is not None}, key=_fold_key)


def training_curve_figure(
    rows: List[Dict[str, Any]],
    selected_tags: Optional[Sequence[str]] = None,
    selected_fold: Optional[str] = None,
    title: str = "Training curves",
):
    """Build a Matplotlib figure for train/validation scalar curves."""
    selected_tags = list(selected_tags or available_training_curve_tags(rows))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    plotted = False
    for tag in selected_tags:
        tag_rows = [row for row in rows if row.get("tag") == tag]
        if selected_fold not in (None, "All"):
            tag_rows = [row for row in tag_rows if str(row.get("fold")) == str(selected_fold)]
        if not tag_rows:
            continue
        # Plot one curve per fold to avoid averaging away instability.
        folds = available_training_curve_folds(tag_rows)
        for fold in folds:
            fold_rows = sorted([row for row in tag_rows if str(row.get("fold")) == fold], key=lambda r: r["step"])
            if not fold_rows:
                continue
            label = f"{tag} / fold {fold}" if selected_fold in (None, "All") else tag
            ax.plot([row["step"] for row in fold_rows], [row["value"] for row in fold_rows], marker="o", linewidth=1.6, markersize=2.5, label=label)
            plotted = True

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.25)
    if plotted:
        ax.legend(fontsize=8, ncols=2)
    else:
        ax.text(0.5, 0.5, "No scalar curves available", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    return fig
