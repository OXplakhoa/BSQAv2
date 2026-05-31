"""
Data Mining Analysis   Entropy, Information Gain, Decision Trees.

Comprehensive DM analysis for stroke classification:
  1. Mutual Information ranking (model-agnostic feature importance)
  2. Shallow Decision Tree with rule visualization
  3. Entropy analysis (class distribution, conditional entropy)
  4. Per-class feature distributions (top features vs stroke type)
  5. Feature correlation heatmap

Part of BSQAv2 Data Mining pillar   addresses lecturer's
"entropy, trees, labeling, verification" requirements.
"""
import sys
from pathlib import Path
_proj_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_proj_root))

import argparse
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import STROKE_TYPES, SEED, K_FOLDS


# -- 1. Mutual Information Ranking ---------------------------------------------

def compute_mutual_information(
    X: np.ndarray, y: np.ndarray, feature_names: List[str]
) -> pd.DataFrame:
    """
    Compute mutual information between each feature and the label.
    MI measures how much knowing a feature reduces uncertainty about the stroke type.
    Model-agnostic   unlike RF importance which depends on the forest structure.
    """
    mi_scores = mutual_info_classif(X, y, random_state=SEED, n_neighbors=5)
    df_mi = pd.DataFrame({
        "feature": feature_names,
        "mutual_information": mi_scores,
    }).sort_values("mutual_information", ascending=False)
    df_mi["rank"] = range(1, len(df_mi) + 1)
    df_mi["normalized"] = df_mi["mutual_information"] / df_mi["mutual_information"].max()
    return df_mi


# -- 2. Entropy Analysis -------------------------------------------------------

def compute_entropy(labels: np.ndarray) -> float:
    """Shannon entropy of label distribution: H(Y) = -sum p(y) * log2(p(y))"""
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs + 1e-10)))


def compute_conditional_entropy(
    X: np.ndarray, y: np.ndarray, feature_idx: int, n_bins: int = 10
) -> float:
    """
    H(Y|X_i): entropy of Y given we know feature i (binned).
    Lower = feature is more informative.
    """
    feature = X[:, feature_idx]
    valid = ~np.isnan(feature)
    if valid.sum() < 10:
        return compute_entropy(y)  # no info

    feature_valid = feature[valid]
    y_valid = y[valid]

    bins = np.percentile(feature_valid, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)
    if len(bins) < 2:
        return compute_entropy(y)

    bin_indices = np.digitize(feature_valid, bins[:-1])

    total = len(y_valid)
    cond_entropy = 0.0
    for b in np.unique(bin_indices):
        mask = bin_indices == b
        if mask.sum() == 0:
            continue
        weight = mask.sum() / total
        cond_entropy += weight * compute_entropy(y_valid[mask])

    return float(cond_entropy)


def information_gain(X: np.ndarray, y: np.ndarray, feature_idx: int) -> float:
    """IG = H(Y) - H(Y|X_i). Higher = more informative."""
    H_y = compute_entropy(y)
    H_y_given_x = compute_conditional_entropy(X, y, feature_idx)
    return H_y - H_y_given_x


# -- 3. Decision Tree ----------------------------------------------------------

def train_shallow_tree(
    X: np.ndarray, y: np.ndarray, feature_names: List[str],
    class_names: List[str], output_dir: Path, max_depth: int = 4,
) -> Dict:
    """
    Train a shallow decision tree and export rules + visualization.
    Shallow tree = interpretable rules the lecturer wants.
    """
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=20,
        random_state=SEED,
        class_weight="balanced",
    )

    # CV accuracy for the tree
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(tree, X, y, cv=skf, scoring="accuracy")
    tree.fit(X, y)
    train_acc = tree.score(X, y)

    # -- Export text rules --
    rules_text = export_text(tree, feature_names=feature_names,
                             max_depth=3, show_weights=True)
    rules_path = output_dir / "decision_tree_rules.txt"
    with open(rules_path, "w") as f:
        f.write(f"Decision Tree Rules (max_depth={max_depth})\n")
        f.write(f"Train Accuracy: {train_acc:.3f}\n")
        f.write(f"CV Accuracy (mean +/- std): {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}\n")
        f.write(f"Entropy H(Y): {compute_entropy(y):.3f} bits\n")
        f.write(f"Max possible IG: {compute_entropy(y):.3f} bits\n")
        f.write("=" * 60 + "\n\n")
        f.write(rules_text)
    print(f"  Saved: {rules_path}")

    # -- Visual tree --
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        tree, feature_names=feature_names,
        class_names=class_names, filled=True, rounded=True,
        proportion=True, fontsize=9, ax=ax,
    )
    ax.set_title(f"Decision Tree (max_depth={max_depth})   "
                 f"Train Acc={train_acc:.3f}, CV Acc={cv_scores.mean():.3f}+/-{cv_scores.std():.3f}")
    fig.tight_layout()
    fig.savefig(output_dir / "decision_tree.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'decision_tree.png'}")

    # -- Feature importance from tree --
    tree_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": tree.feature_importances_,
    }).sort_values("importance", ascending=False)

    return {
        "max_depth": max_depth,
        "train_accuracy": float(train_acc),
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "n_leaves": int(tree.get_n_leaves()),
        "n_nodes": int(tree.tree_.node_count),
        "top_splits": [{"feature": r["feature"], "importance": float(r["importance"])}
                       for r in tree_importance.head(10).to_dict("records")],
    }


# -- 4. Per-Class Feature Distributions ----------------------------------------

def plot_class_distributions(
    df: pd.DataFrame, top_features: List[str], class_names: List[str],
    output_dir: Path, n_features: int = 8,
):
    """Boxplots of top features split by stroke type."""
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.5))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    palette = sns.color_palette("Set2", len(class_names))

    for i, feat in enumerate(top_features[:n_features]):
        ax = axes[i]
        for j, cls in enumerate(class_names):
            vals = df[df["stroke"] == cls][feat].dropna()
            if len(vals) > 0:
                bp = ax.boxplot(
                    [vals], positions=[j], widths=0.6, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 1.5},
                    showfliers=False,
                )
                bp["boxes"][0].set_facecolor(palette[j])
        ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=8)
        ax.set_title(feat[:50], fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Top Features by Stroke Type (Boxplots)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "feature_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'feature_distributions.png'}")


# -- 5. Feature Correlation Matrix ---------------------------------------------

def plot_correlation_matrix(
    X: np.ndarray, feature_names: List[str], output_dir: Path,
    top_n: int = 25,
):
    """Correlation heatmap of top features to identify redundancy."""
    df_mi = compute_mutual_information(X, np.zeros(X.shape[0]), feature_names)
    # Use the top N features by mutual information
    top_indices = [feature_names.index(f) for f in df_mi.head(top_n)["feature"]]

    corr = np.corrcoef(X[:, top_indices].T)
    # Handle NaN correlations
    corr = np.nan_to_num(corr, nan=0)

    # Shorten feature names for display
    short_names = [fn[:35] for fn in np.array(feature_names)[top_indices]]

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=False, cmap="RdBu_r", center=0,
        vmin=-1, vmax=1, square=True, linewidths=0.5,
        xticklabels=short_names, yticklabels=short_names,
        cbar_kws={"shrink": 0.7, "label": "Pearson r"},
        ax=ax,
    )
    ax.set_title(f"Feature Correlation Matrix (Top {top_n} by MI)", fontsize=13)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "feature_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'feature_correlation.png'}")


# -- 6. Entropy Summary --------------------------------------------------------

def entropy_summary(
    X: np.ndarray, y: np.ndarray, feature_names: List[str],
    class_names: List[str], output_dir: Path,
):
    """Compute and save comprehensive entropy analysis."""
    H_y = compute_entropy(y)

    # Per-class entropy contributions
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()

    # IG for top features
    ig_results = []
    for i in range(min(len(feature_names), 30)):
        ig = information_gain(X, y, i)
        ig_results.append({
            "feature": feature_names[i],
            "information_gain": round(ig, 5),
            "normalized_ig": round(ig / H_y, 4) if H_y > 0 else 0,
        })
    ig_results.sort(key=lambda x: x["information_gain"], reverse=True)

    summary = {
        "entropy_H_Y_bits": round(H_y, 4),
        "max_entropy_bits": round(np.log2(len(class_names)), 4),
        "entropy_efficiency": round(H_y / np.log2(len(class_names)), 4),
        "class_distribution": {
            cls: {"count": int(c), "prob": round(float(p), 4),
                  "entropy_contribution": round(-float(p) * np.log2(float(p) + 1e-10), 4)}
            for cls, c, p in zip(class_names, counts, probs)
        },
        "top_information_gains": ig_results[:15],
    }

    with open(output_dir / "entropy_analysis.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {output_dir / 'entropy_analysis.json'}")

    # Print summary
    print(f"\n  Entropy H(Y) = {H_y:.4f} bits (max: {np.log2(len(class_names)):.4f})")
    print(f"  Efficiency: {H_y / np.log2(len(class_names)):.2%}")
    for cls, c, p in zip(class_names, counts, probs):
        contrib = -p * np.log2(p + 1e-10)
        print(f"    {cls:<12}: {int(c):>5} clips ({p:.1%})   contributes {contrib:.3f} bits")
    print(f"\n  Top 5 Information Gains:")
    for ig in ig_results[:5]:
        print(f"    {ig['normalized_ig']:.4f}  {ig['feature']}")

    return summary


# -- Main ----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Data Mining Analysis   Entropy, IG, Decision Trees"
    )
    parser.add_argument("--features", type=str, default="data/biomechanics_features.csv")
    parser.add_argument("--output", type=str, default="results/dm_analysis")
    parser.add_argument("--tree-depth", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Data Mining Analysis   Entropy & Information Gain")
    print("=" * 60)

    # Load
    df = pd.read_csv(args.features)
    feature_cols = [c for c in df.columns if c not in ("id", "stroke")]
    y_raw = df["stroke"].values.astype(str)
    X = df[feature_cols].values.astype(np.float32)

    # Drop NaN
    nan_mask = np.isnan(X).any(axis=1)
    if nan_mask.any():
        print(f"  Dropping {nan_mask.sum()} NaN rows")
        X = X[~nan_mask]
        y_raw = y_raw[~nan_mask]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = list(le.classes_)
    feature_names = feature_cols

    print(f"  Samples: {X.shape[0]} | Features: {X.shape[1]}")
    print(f"  Classes: {dict(zip(class_names, np.bincount(y)))}")

    # -- 1. Mutual Information --
    print(f"\n{'-' * 50}")
    print("  1. Mutual Information Ranking")
    print(f"{'-' * 50}")
    df_mi = compute_mutual_information(X, y, feature_names)
    df_mi.to_csv(output_dir / "mutual_information.csv", index=False)
    print(f"  Saved: {output_dir / 'mutual_information.csv'}")

    # MI bar chart
    top_mi = df_mi.head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(len(top_mi)), top_mi["mutual_information"].values[::-1])
    ax.set_yticks(range(len(top_mi)))
    ax.set_yticklabels(top_mi["feature"].values[::-1], fontsize=8)
    ax.set_xlabel("Mutual Information (bits)")
    ax.set_title("Top 20 Features by Mutual Information with Stroke Type")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "mutual_information.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'mutual_information.png'}")
    print(f"  Top 5: {', '.join(top_mi['feature'].head(5).values)}")

    # -- 2. Entropy Summary --
    print(f"\n{'-' * 50}")
    print("  2. Entropy Analysis")
    print(f"{'-' * 50}")
    entropy_summary(X, y, feature_names, class_names, output_dir)

    # -- 3. Decision Tree --
    print(f"\n{'-' * 50}")
    print(f"  3. Decision Tree (max_depth={args.tree_depth})")
    print(f"{'-' * 50}")
    tree_results = train_shallow_tree(
        X, y, feature_names, class_names, output_dir, args.tree_depth
    )
    with open(output_dir / "decision_tree_results.json", "w") as f:
        json.dump(tree_results, f, indent=2)

    # -- 4. Per-Class Distributions --
    print(f"\n{'-' * 50}")
    print("  4. Per-Class Feature Distributions")
    print(f"{'-' * 50}")
    top_features_mi = df_mi["feature"].head(8).tolist()
    plot_class_distributions(df, top_features_mi, class_names, output_dir)

    # -- 5. Correlation Matrix --
    print(f"\n{'-' * 50}")
    print("  5. Feature Correlation Matrix")
    print(f"{'-' * 50}")
    plot_correlation_matrix(X, feature_names, output_dir)

    print(f"\n{'=' * 60}")
    print(f"  All outputs saved to: {output_dir}/")
    print(f"{'=' * 60}")
    print("Done.")
