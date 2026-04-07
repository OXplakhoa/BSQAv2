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
        Dictionary with mean and std for each scalar metric
    """
    accs = [m["accuracy"] for m in fold_metrics]
    f1s_macro = [m["f1_macro"] for m in fold_metrics]
    f1s_weighted = [m["f1_weighted"] for m in fold_metrics]

    return {
        "accuracy_mean": np.mean(accs),
        "accuracy_std": np.std(accs),
        "f1_macro_mean": np.mean(f1s_macro),
        "f1_macro_std": np.std(f1s_macro),
        "f1_weighted_mean": np.mean(f1s_weighted),
        "f1_weighted_std": np.std(f1s_weighted),
        "n_folds": len(fold_metrics),
    }


def print_fold_summary(fold_metrics: List[Dict], model_name: str = "Model") -> None:
    """Pretty-print aggregated fold results."""
    agg = aggregate_fold_metrics(fold_metrics)
    print(f"\n{'='*50}")
    print(f"  {model_name} — {agg['n_folds']}-Fold CV Results")
    print(f"{'='*50}")
    print(f"  Accuracy:    {agg['accuracy_mean']:.4f} ± {agg['accuracy_std']:.4f}")
    print(f"  F1 (macro):  {agg['f1_macro_mean']:.4f} ± {agg['f1_macro_std']:.4f}")
    print(f"  F1 (weight): {agg['f1_weighted_mean']:.4f} ± {agg['f1_weighted_std']:.4f}")
    print(f"{'='*50}\n")
