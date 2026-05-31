import unittest

import matplotlib.pyplot as plt
import numpy as np

from webapp.components.robustness_viz import (
    degradation_summary_rows,
    degrade_keypoints,
    prediction_delta_rows,
    robustness_curve_figure,
    robustness_curve_rows,
)


class RobustnessVizTests(unittest.TestCase):
    def test_degrade_keypoints_adds_deterministic_noise(self):
        keypoints = np.ones((4, 17, 2), dtype=np.float32)
        a = degrade_keypoints(keypoints, noise_std=0.1, seed=7)
        b = degrade_keypoints(keypoints, noise_std=0.1, seed=7)
        self.assertTrue(np.allclose(a, b))
        self.assertFalse(np.allclose(a, keypoints))

    def test_degrade_keypoints_can_drop_wrist_elbow_and_frames(self):
        keypoints = np.ones((10, 17, 2), dtype=np.float32)
        degraded = degrade_keypoints(
            keypoints,
            drop_wrists=True,
            drop_elbows=True,
            frame_dropout_rate=0.2,
            seed=1,
        )
        self.assertTrue(np.all(degraded[:, [7, 8, 9, 10], :] == 0))
        self.assertGreater(np.sum(np.isclose(degraded, 0.0).all(axis=2).all(axis=1)), 0)

    def test_degradation_summary_rows_report_missing_and_displacement(self):
        original = np.ones((2, 17, 2), dtype=np.float32)
        degraded = original.copy()
        degraded[:, 9] = 0
        rows = degradation_summary_rows(original, degraded)
        metrics = {row["metric"]: row["value"] for row in rows}
        self.assertGreater(metrics["missing_joint_ratio"], 0)
        self.assertGreaterEqual(metrics["mean_coordinate_shift"], 0)

    def test_prediction_delta_rows_compare_probability_changes(self):
        rows = prediction_delta_rows(
            {"smash": 0.7, "clear": 0.3},
            {"smash": 0.4, "clear": 0.6},
        )
        self.assertEqual(rows[0]["class"], "smash")
        self.assertAlmostEqual(rows[0]["delta"], -0.3)

    def test_robustness_curve_rows_are_sorted_by_severity(self):
        rows = robustness_curve_rows([
            {"severity": 0.2, "confidence": 0.5, "prediction": "clear"},
            {"severity": 0.0, "confidence": 0.9, "prediction": "smash"},
        ])
        self.assertEqual(rows[0]["severity"], 0.0)
        self.assertEqual(rows[1]["prediction"], "clear")
    def test_robustness_curve_figure_returns_matplotlib_figure(self):
        fig = robustness_curve_figure([
            {"severity": 0.0, "confidence": 0.9, "prediction": "smash"},
            {"severity": 0.1, "confidence": 0.7, "prediction": "smash"},
        ])
        try:
            self.assertEqual(len(fig.axes), 1)
            self.assertIn("RF confidence", fig.axes[0].get_title())
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
