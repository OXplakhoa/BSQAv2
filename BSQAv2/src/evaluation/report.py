"""Phase 5 evaluation and ablation report generation.

This module assembles report-ready artifacts from completed training/evaluation
outputs.  It does not retrain models; instead it consolidates RF, Decision Tree,
DL CV summaries, curated checkpoints, and Phase 4 quality validation into a
single JSON/CSV/Markdown bundle.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from src.config import PROJECT_ROOT, STROKE_TYPES
from src.observatory.artifacts import ArtifactRegistry
from src.observatory.quality_references import QualityReferenceBank, load_quality_reference_bank, reference_bank_summary
from src.quality.hybrid import HybridQualityScorer


FINAL_DL_METRICS = {
    "model": "GCN + BiLSTM + Attention",
    "family": "Deep Learning",
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
    "note": "Final selected spatial-temporal DL result from PRD/report values.",
}


@dataclass
class EvaluationInputs:
    rf_results_path: Path = PROJECT_ROOT / "results" / "rf_baseline" / "rf_results.json"
    decision_tree_results_path: Path = PROJECT_ROOT / "results" / "dm_analysis" / "decision_tree_results.json"
    dl_cv_summary_path: Path = PROJECT_ROOT / "_colab_results" / "gcn_bilstm_attn_20260528_095136" / "cv_summary.json"
    colab_run_dir: Path = PROJECT_ROOT / "_colab_results" / "gcn_bilstm_attn_20260528_095136"
    artifact_root: Path = PROJECT_ROOT / "webapp" / "artifacts"


def _load_json(path: Path, missing: List[str]) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        missing.append(str(path))
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _float(payload: Dict[str, Any], key: str, default: Optional[float] = None) -> Optional[float]:
    value = payload.get(key, default)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _row_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    return value


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _row_value(row.get(key)) for key in fieldnames})


def rf_model_row(rf_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not rf_results:
        return None
    return {
        "model": "Random Forest",
        "family": "Data Mining",
        "accuracy": _float(rf_results, "accuracy"),
        "accuracy_std": None,
        "f1_macro": _float(rf_results, "f1_macro"),
        "f1_macro_std": None,
        "f1_weighted": _float(rf_results, "f1_weighted"),
        "f1_weighted_std": None,
        "n_samples": rf_results.get("n_samples"),
        "n_features": rf_results.get("n_features"),
        "note": "Strongest numerical classifier in current artifacts.",
    }


def decision_tree_model_row(tree_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not tree_results:
        return None
    return {
        "model": "Decision Tree",
        "family": "Data Mining",
        "accuracy": _float(tree_results, "cv_accuracy_mean"),
        "accuracy_std": _float(tree_results, "cv_accuracy_std"),
        "f1_macro": _float(tree_results, "cv_f1_macro_mean"),
        "f1_macro_std": _float(tree_results, "cv_f1_macro_std"),
        "f1_weighted": None,
        "f1_weighted_std": None,
        "n_samples": tree_results.get("n_samples"),
        "n_features": tree_results.get("n_features"),
        "note": "Interpretable baseline for data-mining discussion.",
    }


def dl_model_rows(local_cv_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = [dict(FINAL_DL_METRICS)]
    rows[0].pop("per_class_f1", None)
    if local_cv_summary:
        rows.append({
            "model": "GCN + BiLSTM + Attention (local checkpoint folder)",
            "family": "Deep Learning",
            "accuracy": _float(local_cv_summary, "accuracy_mean"),
            "accuracy_std": _float(local_cv_summary, "accuracy_std"),
            "f1_macro": _float(local_cv_summary, "f1_macro_mean"),
            "f1_macro_std": _float(local_cv_summary, "f1_macro_std"),
            "f1_weighted": _float(local_cv_summary, "f1_weighted_mean"),
            "f1_weighted_std": _float(local_cv_summary, "f1_weighted_std"),
            "n_samples": None,
            "n_features": None,
            "note": "Exploratory/local checkpoint metrics; kept separate from final report values.",
        })
    return rows


def build_model_comparison(
    rf_results: Dict[str, Any],
    tree_results: Dict[str, Any],
    local_cv_summary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    rows.extend(dl_model_rows(local_cv_summary))
    for row in [rf_model_row(rf_results), decision_tree_model_row(tree_results)]:
        if row:
            rows.append(row)
    return rows


def build_per_class_rows(rf_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    rf_per_class = rf_results.get("per_class", {}) if rf_results else {}
    dl_per_class = FINAL_DL_METRICS["per_class_f1"]
    classes = [cls for cls in STROKE_TYPES if cls in rf_per_class or cls in dl_per_class]
    rows: List[Dict[str, Any]] = []
    for cls in classes:
        rf_metrics = rf_per_class.get(cls, {})
        rows.append({
            "class": cls,
            "rf_precision": _float(rf_metrics, "precision"),
            "rf_recall": _float(rf_metrics, "recall"),
            "rf_f1": _float(rf_metrics, "f1-score"),
            "rf_support": _float(rf_metrics, "support"),
            "dl_f1": dl_per_class.get(cls),
        })
    return rows


def _degrade_sequence(sequence: np.ndarray, noise_std: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    degraded = np.asarray(sequence, dtype=np.float32).copy()
    degraded += rng.normal(0.0, noise_std, size=degraded.shape).astype(np.float32)
    # Slightly dampen wrist/elbow motion to simulate poorer/less decisive motion.
    degraded[:, [7, 8, 9, 10], :] *= 0.9
    return degraded


def run_quality_validation(
    reference_bank: QualityReferenceBank,
    max_per_stroke: int = 2,
    noise_std: float = 0.18,
) -> Dict[str, Any]:
    """Score curated references against degraded copies to validate Phase 4 directionality."""
    scorer = HybridQualityScorer()
    rows: List[Dict[str, Any]] = []
    for stroke in STROKE_TYPES:
        refs = list(reference_bank.get(stroke, []))[:max_per_stroke]
        for idx, (ref_id, sequence) in enumerate(refs):
            baseline = scorer.score(sequence, stroke, references=[(ref_id, sequence)])
            degraded_seq = _degrade_sequence(sequence, noise_std=noise_std, seed=idx + 100)
            degraded = scorer.score(degraded_seq, stroke, references=[(ref_id, sequence)])
            rows.append({
                "stroke_type": stroke,
                "reference_id": ref_id,
                "baseline_quality": baseline["quality_score"],
                "degraded_quality": degraded["quality_score"],
                "baseline_dtw": baseline["dtw_score"],
                "degraded_dtw": degraded["dtw_score"],
                "quality_drop": baseline["quality_score"] - degraded["quality_score"],
                "dtw_drop": (baseline["dtw_score"] or 0.0) - (degraded["dtw_score"] or 0.0),
            })

    drops = [float(row["quality_drop"]) for row in rows]
    dtw_drops = [float(row["dtw_drop"]) for row in rows]
    pass_rate = float(np.mean([drop > 0 for drop in drops])) if drops else None
    return {
        "rows": rows,
        "summary": {
            "n_pairs": len(rows),
            "mean_quality_drop": float(np.mean(drops)) if drops else None,
            "mean_dtw_drop": float(np.mean(dtw_drops)) if dtw_drops else None,
            "pass_rate_quality_drop_positive": pass_rate,
            "noise_std": noise_std,
        },
    }


def checkpoint_inventory(colab_run_dir: Path) -> Dict[str, Any]:
    colab_run_dir = Path(colab_run_dir)
    checkpoints = sorted(path.name for path in colab_run_dir.glob("best_model_fold*.pth")) if colab_run_dir.exists() else []
    event_files = sorted(str(path.relative_to(colab_run_dir)) for path in colab_run_dir.glob("fold_*/events.out.tfevents.*")) if colab_run_dir.exists() else []
    return {
        "run_dir": str(colab_run_dir),
        "checkpoints": checkpoints,
        "tensorboard_event_files": event_files,
        "n_checkpoints": len(checkpoints),
        "n_event_files": len(event_files),
    }


def artifact_inventory(inputs: EvaluationInputs) -> Dict[str, Any]:
    return {
        "rf_results": str(inputs.rf_results_path),
        "decision_tree_results": str(inputs.decision_tree_results_path),
        "dl_cv_summary": str(inputs.dl_cv_summary_path),
        "colab_run_dir": str(inputs.colab_run_dir),
        "artifact_root": str(inputs.artifact_root),
    }


def _write_accuracy_plot(path: Path, rows: List[Dict[str, Any]]) -> Optional[str]:
    plot_rows = [row for row in rows if row.get("accuracy") is not None]
    if not plot_rows:
        return None
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    labels = [row["model"] for row in plot_rows]
    values = [float(row["accuracy"]) for row in plot_rows]
    fig, ax = plt.subplots(figsize=(9, max(3, 0.6 * len(labels))))
    ax.barh(labels, values, color="#4f81bd")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Accuracy")
    ax.set_title("BSQAv2 model accuracy comparison")
    for idx, value in enumerate(values):
        ax.text(min(value + 0.01, 0.98), idx, f"{value:.3f}", va="center")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def _markdown_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    if not rows:
        return "_No rows available._\n"
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col)
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append("" if value is None else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _write_markdown_report(path: Path, summary: Dict[str, Any]) -> None:
    comparison = summary.get("model_comparison", [])
    per_class = summary.get("per_class_metrics", [])
    quality_summary = summary.get("quality_validation", {}).get("summary", {})
    inventory = summary.get("checkpoint_inventory", {})

    content = f"""# BSQAv2 Phase 5 Evaluation Report

Generated at: `{summary.get('generated_at')}`

## Model comparison

{_markdown_table(comparison, ['model', 'family', 'accuracy', 'accuracy_std', 'f1_macro', 'f1_weighted', 'note'])}

## Per-class metrics

{_markdown_table(per_class, ['class', 'rf_precision', 'rf_recall', 'rf_f1', 'rf_support', 'dl_f1'])}

## Quality-score validation

Phase 4 quality scoring was sanity-checked by comparing curated reference skeletons with artificially degraded variants.

- Pairs: `{quality_summary.get('n_pairs')}`
- Mean quality-score drop: `{quality_summary.get('mean_quality_drop')}`
- Mean DTW-score drop: `{quality_summary.get('mean_dtw_drop')}`
- Positive-drop pass rate: `{quality_summary.get('pass_rate_quality_drop_positive')}`

## Checkpoint inventory

- Checkpoints found: `{inventory.get('n_checkpoints')}`
- TensorBoard event files found: `{inventory.get('n_event_files')}`
- Run directory: `{inventory.get('run_dir')}`

## Conclusions

1. Random Forest remains the strongest numerical classifier in current local artifacts.
2. GCN + BiLSTM + Attention is retained as the proposed spatial-temporal DL architecture and supports attention inspection.
3. Phase 4 quality scoring is heuristic and should be presented as 2D-pose technique similarity, not expert-labeled coaching truth.
4. Custom Upload is beta; curated cached mode is the stable defense path.
"""
    path.write_text(content, encoding="utf-8")


def generate_evaluation_report(
    output_dir: Path,
    inputs: Optional[EvaluationInputs] = None,
    include_quality_validation: bool = True,
    max_quality_refs_per_stroke: int = 2,
) -> Dict[str, Any]:
    inputs = inputs or EvaluationInputs()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    missing_artifacts: List[str] = []

    rf_results = _load_json(inputs.rf_results_path, missing_artifacts)
    tree_results = _load_json(inputs.decision_tree_results_path, missing_artifacts)
    local_cv_summary = _load_json(inputs.dl_cv_summary_path, missing_artifacts)

    comparison = build_model_comparison(rf_results, tree_results, local_cv_summary)
    per_class = build_per_class_rows(rf_results)
    checkpoints = checkpoint_inventory(inputs.colab_run_dir)

    quality_validation = {"rows": [], "summary": {"n_pairs": 0}}
    reference_summary: Dict[str, int] = {}
    if include_quality_validation:
        registry = ArtifactRegistry(root=inputs.artifact_root)
        bank = load_quality_reference_bank(registry=registry, max_per_stroke=max_quality_refs_per_stroke)
        reference_summary = reference_bank_summary(bank)
        quality_validation = run_quality_validation(bank, max_per_stroke=max_quality_refs_per_stroke)

    generated_files: Dict[str, str] = {}
    comparison_csv = output_dir / "model_comparison.csv"
    per_class_csv = output_dir / "per_class_metrics.csv"
    quality_csv = output_dir / "quality_validation.csv"
    _write_csv(comparison_csv, comparison)
    _write_csv(per_class_csv, per_class)
    _write_csv(quality_csv, quality_validation.get("rows", []))
    generated_files.update({
        "model_comparison_csv": str(comparison_csv),
        "per_class_metrics_csv": str(per_class_csv),
        "quality_validation_csv": str(quality_csv),
    })

    plot_path = _write_accuracy_plot(output_dir / "model_accuracy_comparison.png", comparison)
    if plot_path:
        generated_files["model_accuracy_comparison_png"] = plot_path

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact_inventory": artifact_inventory(inputs),
        "missing_artifacts": missing_artifacts,
        "model_comparison": comparison,
        "per_class_metrics": per_class,
        "quality_reference_summary": reference_summary,
        "quality_validation": quality_validation,
        "checkpoint_inventory": checkpoints,
        "generated_files": generated_files,
        "conclusions": [
            "Random Forest is strongest numerically in current artifacts.",
            "GCN + BiLSTM + Attention remains the proposed spatial-temporal architecture.",
            "Phase 4 quality scoring is heuristic 2D-pose technique similarity.",
            "Curated cached demo is the stable defense path; Custom Upload is beta.",
        ],
    }

    summary_json = output_dir / "evaluation_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    generated_files["evaluation_summary_json"] = str(summary_json)

    markdown_path = output_dir / "evaluation_report.md"
    _write_markdown_report(markdown_path, summary)
    generated_files["evaluation_report_md"] = str(markdown_path)

    # Persist generated file list after all files are known.
    summary["generated_files"] = generated_files
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
