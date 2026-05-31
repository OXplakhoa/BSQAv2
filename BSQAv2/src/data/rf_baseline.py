"""
Random Forest baseline for stroke classification.

Loads biomechanical features and trains a Random Forest with
5-fold stratified CV. Outputs confusion matrix, per-class metrics,
and feature importance ranking.

Part of BSQAv2 Data Mining pillar.
"""
import sys
from pathlib import Path
_proj_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_proj_root))

import argparse
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score,
)
from sklearn.preprocessing import LabelEncoder

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import STROKE_TYPES, SEED, K_FOLDS


def load_features(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Load feature CSV and return X, y, y_encoded, feature_names.

    Drops 'id' and 'stroke' columns; stroke column is encoded via LabelEncoder.
    Rows with NaN features are dropped with a warning.
    """
    df = pd.read_csv(csv_path)

    # Separate features and labels
    feature_cols = [c for c in df.columns if c not in ("id", "stroke")]
    y_raw = df["stroke"].values.astype(str)  # ensure string for sklearn
    X = df[feature_cols].values.astype(np.float32)

    # Drop rows with NaN features
    nan_mask = np.isnan(X).any(axis=1)
    if nan_mask.any():
        print(f"  Dropping {nan_mask.sum()} clips with NaN features "
              f"({nan_mask.sum() / len(df) * 100:.1f}%)")
        X = X[~nan_mask]
        y_raw = y_raw[~nan_mask]

    # Label encode stroke types
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    feature_names = feature_cols

    class_names = list(le.classes_)
    print(f"  Loaded {X.shape[0]} clips, {X.shape[1]} features")
    print(f"  Classes: {dict(zip(class_names, range(len(class_names))))}")
    print(f"  Distribution: {dict(zip(*np.unique(y_raw, return_counts=True)))}")

    return X, y, y_raw, feature_names, class_names


def export_rf_artifact(
    model: RandomForestClassifier,
    label_encoder: LabelEncoder,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    class_names: list,
    output_dir: Path,
    metadata: Optional[Dict] = None,
) -> Path:
    """
    Save a loadable Random Forest bundle for Streamlit inference.

    The bundle includes the fitted model plus the exact feature order, label
    encoder, per-feature medians for NaN imputation, and per-class feature
    averages for local explanations.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    medians = np.nanmedian(X, axis=0)
    feature_medians = {
        name: float(value) if np.isfinite(value) else 0.0
        for name, value in zip(feature_names, medians)
    }

    class_averages = {}
    for class_index, class_name in enumerate(class_names):
        class_X = X[y == class_index]
        if len(class_X) == 0:
            class_averages[class_name] = {name: 0.0 for name in feature_names}
            continue
        means = np.nanmean(class_X, axis=0)
        class_averages[class_name] = {
            name: float(value) if np.isfinite(value) else feature_medians[name]
            for name, value in zip(feature_names, means)
        }

    bundle = {
        "schema_version": 1,
        "model": model,
        "label_encoder": label_encoder,
        "feature_names": list(feature_names),
        "feature_medians": feature_medians,
        "class_averages": class_averages,
        "metadata": metadata or {},
    }

    artifact_path = output_dir / "rf_model_bundle.joblib"
    joblib.dump(bundle, artifact_path)

    manifest = {
        "schema_version": 1,
        "artifact": artifact_path.name,
        "n_features": len(feature_names),
        "feature_names": list(feature_names),
        "class_names": list(class_names),
        "metadata": metadata or {},
    }
    with open(output_dir / "rf_model_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"  Saved RF artifact: {artifact_path}")
    print(f"  Saved RF manifest: {output_dir / 'rf_model_manifest.json'}")
    return artifact_path


def train_rf_cv(
    X: np.ndarray,
    y: np.ndarray,
    y_raw: np.ndarray,
    feature_names: list,
    class_names: list,
    output_dir: Path,
    n_estimators: int = 200,
    max_depth: int = 15,
    export_artifact_dir: Optional[Path] = None,
    source_features: Optional[str] = None,
) -> Dict:
    """
    Train Random Forest with stratified 5-fold CV.
    Saves confusion matrix and feature importance plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=5,
        random_state=SEED,
        n_jobs=-1,
        class_weight="balanced",
    )

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

    # Cross-validated predictions
    print(f"\n  Running {K_FOLDS}-fold stratified CV...")
    y_pred = cross_val_predict(rf, X, y, cv=skf, n_jobs=-1)
    y_proba = cross_val_predict(rf, X, y, cv=skf, method="predict_proba", n_jobs=-1)

    # -- Metrics --
    acc = accuracy_score(y, y_pred)
    f1_macro = f1_score(y, y_pred, average="macro")
    f1_weighted = f1_score(y, y_pred, average="weighted")

    # Per-class metrics
    report = classification_report(y, y_pred, target_names=class_names, output_dict=True)

    print(f"\n  Accuracy:     {acc:.4f}")
    print(f"  F1 (macro):   {f1_macro:.4f}")
    print(f"  F1 (weighted):{f1_weighted:.4f}")
    print(f"\n{classification_report(y, y_pred, target_names=class_names)}")

    # -- Confusion matrix --
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Random Forest - Confusion Matrix\n"
                 f"Accuracy={acc:.3f} | F1-macro={f1_macro:.3f}")
    fig.tight_layout()
    fig.savefig(output_dir / "rf_confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'rf_confusion_matrix.png'}")

    # -- Normalized confusion matrix --
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Random Forest - Normalized Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "rf_confusion_matrix_norm.png", dpi=150)
    plt.close(fig)

    # -- Feature importance --
    # Fit on full data for importance
    rf_full = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=5, random_state=SEED, n_jobs=-1,
        class_weight="balanced",
    )
    rf_full.fit(X, y)

    importances = rf_full.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Top 30 features
    top_n = min(30, len(feature_names))
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(top_n), importances[indices[:top_n]][::-1], align="center")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices[:top_n][::-1]])
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title(f"Random Forest - Top {top_n} Feature Importances")
    fig.tight_layout()
    fig.savefig(output_dir / "rf_feature_importance.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'rf_feature_importance.png'}")

    # -- Save results --
    results = {
        "model": "RandomForest",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "k_folds": K_FOLDS,
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "per_class": {
            cls: {k: float(v) for k, v in metrics.items() if k in ("precision", "recall", "f1-score", "support")}
            for cls, metrics in report.items()
            if cls not in ("accuracy", "macro avg", "weighted avg")
        },
        "top_features": [
            {"rank": i + 1, "feature": feature_names[idx], "importance": float(importances[idx])}
            for i, idx in enumerate(indices[:20])
        ],
    }
    with open(output_dir / "rf_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {output_dir / 'rf_results.json'}")

    if export_artifact_dir is not None:
        export_rf_artifact(
            model=rf_full,
            label_encoder=LabelEncoder().fit(y_raw),
            X=X,
            y=y,
            feature_names=feature_names,
            class_names=class_names,
            output_dir=Path(export_artifact_dir),
            metadata={
                "source_features": source_features,
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_leaf": 5,
                "class_weight": "balanced",
                "random_state": SEED,
                "n_samples": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "nan_strategy": "drop rows during training; median-impute during inference",
            },
        )

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Random Forest baseline for stroke classification")
    parser.add_argument("--features", type=str, default="data/biomechanics_features.csv",
                        help="Path to biomechanics features CSV")
    parser.add_argument("--output", type=str, default="results/rf_baseline",
                        help="Output directory for plots and results")
    parser.add_argument("--n-estimators", type=int, default=200,
                        help="Number of trees in Random Forest")
    parser.add_argument("--max-depth", type=int, default=15,
                        help="Maximum tree depth")
    parser.add_argument("--export-artifact", action="store_true",
                        help="Export fitted RF bundle for Streamlit inference")
    parser.add_argument("--artifact-output", type=str,
                        default="webapp/artifacts/models/rf_baseline",
                        help="Directory for exported RF inference bundle")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = Path(args.output)

    print("=" * 60)
    print("  Random Forest - Stroke Classification Baseline")
    print("=" * 60)

    X, y, y_raw, feature_names, class_names = load_features(args.features)
    results = train_rf_cv(
        X, y, y_raw, feature_names, class_names, output_dir,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        export_artifact_dir=Path(args.artifact_output) if args.export_artifact else None,
        source_features=args.features,
    )

    # -- Quick per-class summary --
    print(f"\n{'-' * 50}")
    print(f"  Per-Class F1 Scores")
    print(f"{'-' * 50}")
    for cls_name, metrics in results["per_class"].items():
        print(f"  {cls_name:<12} F1={metrics['f1-score']:.4f}  "
              f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  "
              f"(n={int(metrics['support'])})")

    print(f"\n  Top 5 features:")
    for feat in results["top_features"][:5]:
        print(f"    {feat['rank']}. {feat['feature']:<40} {feat['importance']:.4f}")

    print("\nDone.")
