import tempfile
import unittest
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from webapp.components.eval_viz import (
    available_training_curve_folds,
    available_training_curve_tags,
    comparison_rows,
    discover_training_run_dirs,
    dl_final_metrics,
    fold_artifact_rows,
    load_tensorboard_scalars,
    per_class_metric_rows,
    rf_summary_metrics,
    training_curve_figure,
)


class EvalVizTests(unittest.TestCase):
    def test_rf_summary_metrics_extract_main_scores(self):
        metrics = rf_summary_metrics({"accuracy": 0.7, "f1_macro": 0.6, "f1_weighted": 0.65})
        self.assertEqual(metrics["accuracy"], 0.7)
        self.assertEqual(metrics["f1_macro"], 0.6)
        self.assertEqual(metrics["f1_weighted"], 0.65)

    def test_dl_final_metrics_use_defense_values(self):
        metrics = dl_final_metrics()
        self.assertAlmostEqual(metrics["accuracy"], 0.6563)
        self.assertIn("per_class_f1", metrics)
        self.assertEqual(metrics["per_class_f1"]["lift"], 0.434)

    def test_comparison_rows_include_rf_dl_and_decision_tree_when_available(self):
        rows = comparison_rows(
            rf_results={"accuracy": 0.71, "f1_macro": 0.70, "f1_weighted": 0.72},
            decision_tree_results={"cv_accuracy_mean": 0.54, "cv_accuracy_std": 0.01},
            dl_metrics={"accuracy": 0.65, "accuracy_std": 0.03, "f1_macro": 0.64, "f1_weighted": 0.66},
        )
        names = [row["model"] for row in rows]
        self.assertIn("Random Forest", names)
        self.assertIn("GCN + BiLSTM + Attention", names)
        self.assertIn("Decision Tree", names)

    def test_per_class_metric_rows_extract_rf_and_dl_values(self):
        rows = per_class_metric_rows(
            rf_results={"per_class": {"smash": {"f1-score": 0.9}, "lift": {"f1-score": 0.6}}},
            dl_metrics={"per_class_f1": {"smash": 0.824, "lift": 0.434}},
        )
        lift = [row for row in rows if row["class"] == "lift"][0]
        self.assertEqual(lift["rf_f1"], 0.6)
        self.assertEqual(lift["dl_f1"], 0.434)

    def test_fold_artifact_rows_parse_summary_and_checkpoint_names(self):
        rows = fold_artifact_rows(
            {"n_folds": 2, "accuracy_mean": 0.6, "accuracy_std": 0.1},
            ["best_model_fold0.pth", "best_model_fold1.pth"],
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["fold"], 0)
        self.assertEqual(rows[0]["checkpoint"], "best_model_fold0.pth")
        self.assertEqual(rows[0]["cv_accuracy_mean"], 0.6)

    def test_load_tensorboard_scalars_reads_train_val_curves(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "gcn_test_run"
            fold_dir = run_dir / "fold_0"
            writer = SummaryWriter(log_dir=str(fold_dir))
            writer.add_scalar("Loss/train", 1.0, 0)
            writer.add_scalar("Loss/val", 1.2, 0)
            writer.add_scalar("Acc/train", 0.5, 0)
            writer.add_scalar("Acc/val", 0.4, 0)
            writer.flush()
            writer.close()

            runs = discover_training_run_dirs(Path(tmp))
            self.assertIn(run_dir, runs)

            rows = load_tensorboard_scalars(run_dir)
            tags = available_training_curve_tags(rows)
            folds = available_training_curve_folds(rows)
            self.assertIn("Loss/train", tags)
            self.assertIn("Acc/val", tags)
            self.assertEqual(folds, ["0"])
            self.assertTrue(any(row["value"] == 1.0 for row in rows))

    def test_training_curve_figure_returns_matplotlib_figure(self):
        rows = [
            {"fold": "0", "tag": "Loss/train", "step": 0, "value": 1.0},
            {"fold": "0", "tag": "Loss/train", "step": 1, "value": 0.8},
        ]
        fig = training_curve_figure(rows, selected_tags=["Loss/train"], selected_fold="0")
        self.assertTrue(hasattr(fig, "savefig"))


if __name__ == "__main__":
    unittest.main()
