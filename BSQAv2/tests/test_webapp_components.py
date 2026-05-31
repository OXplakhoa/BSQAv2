import unittest

import matplotlib.pyplot as plt
import numpy as np

from src.observatory.schema import PipelineRun, PredictionResult
from webapp.components.charts import (
    attention_frame_importance,
    critical_visibility_rows,
    missing_joint_ratio_rows,
    probability_rows,
    skeleton_frame_rows,
)
from webapp.components.skeleton_view import skeleton_figure


def _run():
    run = PipelineRun.new("run_1", "sample_1", "curated")
    run.rf_prediction = PredictionResult.from_probabilities({"clear": 0.2, "smash": 0.8})
    run.pose_qc = {
        "missing_joint_ratio_by_frame": [0.0, 0.25],
        "critical_joint_visibility": {"left_wrist": 0.4, "right_wrist": 0.9},
    }
    run.arrays["raw_keypoints"] = np.zeros((2, 17, 2), dtype=np.float32)
    run.arrays["raw_keypoints"][1, 9] = [100.0, 200.0]
    run.arrays["normalized_keypoints"] = np.zeros((64, 17, 2), dtype=np.float32)
    run.arrays["attention_weights"] = np.ones((64, 64), dtype=np.float32) / 64.0
    return run


class WebappChartDataTests(unittest.TestCase):
    def test_probability_rows_are_sorted_descending(self):
        rows = probability_rows({"clear": 0.2, "smash": 0.8})
        self.assertEqual(rows[0]["class"], "smash")
        self.assertEqual(rows[1]["class"], "clear")

    def test_pose_chart_rows_are_extracted_from_pipeline_run(self):
        run = _run()
        self.assertEqual(len(missing_joint_ratio_rows(run)), 2)
        self.assertEqual(critical_visibility_rows(run)[0]["joint"], "left_wrist")

    def test_skeleton_frame_rows_include_critical_joint_marker(self):
        rows = skeleton_frame_rows(_run(), frame_index=1)
        wrist = [row for row in rows if row["joint"] == "left_wrist"][0]
        self.assertEqual(wrist["x"], 100.0)
        self.assertTrue(wrist["critical"])

    def test_attention_frame_importance_returns_64_frames(self):
        rows = attention_frame_importance(_run())
        self.assertEqual(len(rows), 64)
        self.assertAlmostEqual(rows[0]["attention"], 1 / 64)

    def test_skeleton_figure_returns_matplotlib_figure(self):
        run = _run()
        fig = skeleton_figure(run.arrays["raw_keypoints"], frame_index=1)
        try:
            self.assertEqual(len(fig.axes), 1)
            self.assertIn("Skeleton frame", fig.axes[0].get_title())
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
