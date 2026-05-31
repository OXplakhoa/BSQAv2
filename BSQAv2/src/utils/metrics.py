"""
Classification metrics for evaluation — v2
Wrappers around sklearn for consistent reporting across models/folds.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compute classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Optional list of class names for the report

    Returns:
        Dictionary with accuracy, f1_macro, f1_weighted, confusion_matrix,
        and full classification_report string
    """
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def aggregate_fold_metrics(fold_metrics: List[Dict]) -> Dict:
    """
    Aggregate metrics across K folds (mean ± std).

    Args:
        fold_metrics: List of metric dicts from compute_metrics()

    Returns:
        Dictionary with mean and std for each scalar metric, plus per-class F1
    """
    accs = [m["accuracy"] for m in fold_metrics]
    f1s_macro = [m["f1_macro"] for m in fold_metrics]
    f1s_weighted = [m["f1_weighted"] for m in fold_metrics]

    # Extract per-class F1 from classification report
    # The report has lines like: "      smash       0.87      0.85      0.86       158"
    from src.config import STROKE_TYPES
    per_class_f1s = {cls: [] for cls in STROKE_TYPES}
    for m in fold_metrics:
        report = m.get("classification_report", "")
        for cls_name in STROKE_TYPES:
            for line in report.split('\n'):
                if line.strip().startswith(cls_name):
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        try:
                            per_class_f1s[cls_name].append(float(parts[2]))
                        except ValueError:
                            pass
                    break

    per_class = {}
    for cls_name in STROKE_TYPES:
        vals = per_class_f1s[cls_name]
        if vals:
            per_class[f"f1_{cls_name}_mean"] = float(np.mean(vals))
            per_class[f"f1_{cls_name}_std"] = float(np.std(vals))

    # Sum confusion matrices
    cm_sum = np.sum([m["confusion_matrix"] for m in fold_metrics], axis=0)

    return {
        "accuracy_mean": np.mean(accs),
        "accuracy_std": np.std(accs),
        "f1_macro_mean": np.mean(f1s_macro),
        "f1_macro_std": np.std(f1s_macro),
        "f1_weighted_mean": np.mean(f1s_weighted),
        "f1_weighted_std": np.std(f1s_weighted),
        "n_folds": len(fold_metrics),
        "per_class_f1": per_class,
        "confusion_matrix_sum": cm_sum.tolist(),
    }


def print_fold_summary(fold_metrics: List[Dict], model_name: str = "Model") -> None:
    """Pretty-print aggregated fold results with per-class F1."""
    agg = aggregate_fold_metrics(fold_metrics)
    print(f"\n{'='*50}")
    print(f"  {model_name} — {agg['n_folds']}-Fold CV Results")
    print(f"{'='*50}")
    print(f"  Accuracy:    {agg['accuracy_mean']:.4f} ± {agg['accuracy_std']:.4f}")
    print(f"  F1 (macro):  {agg['f1_macro_mean']:.4f} ± {agg['f1_macro_std']:.4f}")
    print(f"  F1 (weight): {agg['f1_weighted_mean']:.4f} ± {agg['f1_weighted_std']:.4f}")
    
    per_class = agg.get("per_class_f1", {})
    if per_class:
        print(f"\n  Per-class F1 (mean ± std):")
        from src.config import STROKE_TYPES
        for cls_name in STROKE_TYPES:
            mean_k = f"f1_{cls_name}_mean"
            std_k = f"f1_{cls_name}_std"
            if mean_k in per_class:
                print(f"    {cls_name:<12s}: {per_class[mean_k]:.3f} ± {per_class[std_k]:.3f}")
    print(f"{'='*50}\n")
