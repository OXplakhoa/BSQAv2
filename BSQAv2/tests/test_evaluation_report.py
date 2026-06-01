import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.evaluation.report import (
    EvaluationInputs,
    build_model_comparison,
    build_per_class_rows,
    generate_evaluation_report,
    run_quality_validation,
)
from src.observatory.artifacts import ArtifactRegistry
from src.observatory.schema import PipelineRun, save_pipeline_run


class EvaluationReportTests(unittest.TestCase):
    def test_build_model_comparison_contains_rf_and_final_dl(self):
        rf = {"accuracy": 0.7, "f1_macro": 0.6, "f1_weighted": 0.65, "n_samples": 10, "n_features": 3}
        rows = build_model_comparison(rf, {}, {})
        names = [row["model"] for row in rows]
        self.assertIn("GCN + BiLSTM + Attention", names)
        self.assertIn("Random Forest", names)

    def test_per_class_rows_merge_rf_and_dl_metrics(self):
        rf = {
            "per_class": {
                "smash": {"precision": 0.8, "recall": 0.7, "f1-score": 0.75, "support": 5},
            }
        }
        rows = build_per_class_rows(rf)
        smash = next(row for row in rows if row["class"] == "smash")
        self.assertEqual(smash["rf_f1"], 0.75)
        self.assertIsNotNone(smash["dl_f1"])

    def test_quality_validation_scores_degraded_lower(self):
        seq = np.zeros((64, 17, 2), dtype=np.float32)
        seq[:, 10, 0] = np.linspace(0, 1, 64)
        seq[:, 10, 1] = np.linspace(-1, -2, 64)
        bank = {"smash": [("ref", seq)]}
        result = run_quality_validation(bank, max_per_stroke=1, noise_std=0.25)
        self.assertEqual(result["summary"]["n_pairs"], 1)
        self.assertGreater(result["rows"][0]["dtw_drop"], 0)

    def test_generate_evaluation_report_writes_json_csv_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rf_path = root / "rf_results.json"
            tree_path = root / "tree.json"
            dl_path = root / "cv_summary.json"
            colab_dir = root / "colab"
            artifact_root = root / "artifacts"
            output_dir = root / "out"

            rf_path.write_text(json.dumps({
                "accuracy": 0.72,
                "f1_macro": 0.71,
                "f1_weighted": 0.72,
                "n_samples": 100,
                "n_features": 5,
                "per_class": {
                    "smash": {"precision": 0.9, "recall": 0.8, "f1-score": 0.85, "support": 20}
                },
            }), encoding="utf-8")
            tree_path.write_text(json.dumps({"cv_accuracy_mean": 0.5, "cv_accuracy_std": 0.1}), encoding="utf-8")
            dl_path.write_text(json.dumps({"accuracy_mean": 0.6, "accuracy_std": 0.05, "f1_macro_mean": 0.58, "n_folds": 5}), encoding="utf-8")
            colab_dir.mkdir()
            (colab_dir / "best_model_fold0.pth").write_bytes(b"fake")

            registry = ArtifactRegistry(root=artifact_root)
            registry.ensure_layout()
            run = PipelineRun.new("run1", "sample1", "skeleton", ground_truth="smash")
            run.pose_qc = {"reliability_score": 1.0}
            arr = np.zeros((64, 17, 2), dtype=np.float32)
            arr[:, 10, 0] = np.linspace(0, 1, 64)
            run.arrays["normalized_keypoints"] = arr
            save_pipeline_run(run, registry.pipeline_runs_dir / "run1")
            registry.curated_manifest.write_text(json.dumps({
                "samples": [{
                    "sample_id": "sample1",
                    "title": "Sample",
                    "stroke_type": "smash",
                    "ground_truth": "smash",
                    "video_path": "data/clips/smash/sample.mp4",
                    "pipeline_run_dir": "pipeline_runs/run1",
                }]
            }), encoding="utf-8")

            summary = generate_evaluation_report(
                output_dir=output_dir,
                inputs=EvaluationInputs(
                    rf_results_path=rf_path,
                    decision_tree_results_path=tree_path,
                    dl_cv_summary_path=dl_path,
                    colab_run_dir=colab_dir,
                    artifact_root=artifact_root,
                ),
                include_quality_validation=True,
            )

            self.assertTrue((output_dir / "evaluation_summary.json").exists())
            self.assertTrue((output_dir / "model_comparison.csv").exists())
            self.assertTrue((output_dir / "per_class_metrics.csv").exists())
            self.assertTrue((output_dir / "quality_validation.csv").exists())
            self.assertTrue((output_dir / "evaluation_report.md").exists())
            self.assertEqual(summary["quality_validation"]["summary"]["n_pairs"], 1)


if __name__ == "__main__":
    unittest.main()
