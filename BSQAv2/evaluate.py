"""BSQAv2 Phase 5 evaluation/report generator.

Examples:

    python evaluate.py --output results/evaluation_report --all-models --kfold 5
    python evaluate.py --output results/evaluation_report --skip-quality-validation

The script consolidates existing artifacts; it does not retrain models.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import PROJECT_ROOT
from src.evaluation.report import EvaluationInputs, generate_evaluation_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BSQAv2 Phase 5 evaluation report")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "evaluation_report", help="Directory to save evaluation report artifacts")
    parser.add_argument("--all-models", action="store_true", help="Kept for plan compatibility; report includes all available model artifacts")
    parser.add_argument("--kfold", type=int, default=5, help="Expected number of folds for report metadata")
    parser.add_argument("--rf-results", type=Path, default=PROJECT_ROOT / "results" / "rf_baseline" / "rf_results.json")
    parser.add_argument("--decision-tree-results", type=Path, default=PROJECT_ROOT / "results" / "dm_analysis" / "decision_tree_results.json")
    parser.add_argument("--dl-cv-summary", type=Path, default=PROJECT_ROOT / "_colab_results" / "gcn_bilstm_attn_20260528_095136" / "cv_summary.json")
    parser.add_argument("--colab-run-dir", type=Path, default=PROJECT_ROOT / "_colab_results" / "gcn_bilstm_attn_20260528_095136")
    parser.add_argument("--artifact-root", type=Path, default=PROJECT_ROOT / "webapp" / "artifacts")
    parser.add_argument("--manual-quality-labels", type=Path, default=PROJECT_ROOT / "data" / "manual_quality_labels_50.csv")
    parser.add_argument("--manual-quality-results", type=Path, default=PROJECT_ROOT / "data" / "manual_quality_evaluation_50.csv")
    parser.add_argument("--skip-quality-validation", action="store_true", help="Skip Phase 4 reference-vs-degraded quality validation")
    parser.add_argument("--max-quality-refs-per-stroke", type=int, default=2, help="Maximum curated references per stroke used for quality validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = EvaluationInputs(
        rf_results_path=args.rf_results,
        decision_tree_results_path=args.decision_tree_results,
        dl_cv_summary_path=args.dl_cv_summary,
        colab_run_dir=args.colab_run_dir,
        artifact_root=args.artifact_root,
        manual_quality_labels_path=args.manual_quality_labels,
        manual_quality_results_path=args.manual_quality_results,
    )
    summary = generate_evaluation_report(
        output_dir=args.output,
        inputs=inputs,
        include_quality_validation=not args.skip_quality_validation,
        max_quality_refs_per_stroke=args.max_quality_refs_per_stroke,
    )

    print("BSQAv2 Phase 5 evaluation report generated")
    print(f"Output: {Path(args.output).resolve()}")
    print(f"Models: {len(summary['model_comparison'])}")
    print(f"Per-class rows: {len(summary['per_class_metrics'])}")
    q_summary = summary.get("quality_validation", {}).get("summary", {})
    print(f"Quality validation pairs: {q_summary.get('n_pairs', 0)}")
    if summary.get("missing_artifacts"):
        print("Missing artifacts:")
        for item in summary["missing_artifacts"]:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
