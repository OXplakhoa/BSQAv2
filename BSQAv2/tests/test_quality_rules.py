import unittest

import numpy as np

from src.quality.rules import evaluate_stroke_rules, rule_score_summary


def _base_sequence():
    seq = np.zeros((64, 17, 2), dtype=np.float32)
    # hips
    seq[:, 11] = [-0.2, 0.0]
    seq[:, 12] = [0.2, 0.0]
    # shoulders
    seq[:, 5] = [-0.3, -1.0]
    seq[:, 6] = [0.3, -1.0]
    # right arm motion
    seq[:, 8] = [0.6, -1.2]
    seq[:, 10] = [0.8, -1.4]
    seq[:, 10, 0] += np.linspace(0, 1.5, 64)
    seq[:, 10, 1] += np.sin(np.linspace(0, np.pi, 64)) * -0.5
    # left arm
    seq[:, 7] = [-0.6, -1.0]
    seq[:, 9] = [-0.8, -1.1]
    # knees/ankles
    seq[:, 13] = [-0.2, 0.8]
    seq[:, 14] = [0.2, 0.8]
    seq[:, 15] = [-0.2, 1.5]
    seq[:, 16] = [0.2, 1.5]
    return seq


class QualityRulesTests(unittest.TestCase):
    def test_smash_rules_return_scores_and_feedback(self):
        result = evaluate_stroke_rules(_base_sequence(), "smash")
        self.assertEqual(result.stroke_type, "smash")
        self.assertGreaterEqual(len(result.rule_results), 3)
        self.assertGreaterEqual(result.overall_score, 0)
        self.assertLessEqual(result.overall_score, 100)
        self.assertTrue(result.feedback)

    def test_all_stroke_types_are_supported(self):
        for stroke in ["smash", "clear", "drop_shot", "net_shot", "lift"]:
            with self.subTest(stroke=stroke):
                result = evaluate_stroke_rules(_base_sequence(), stroke)
                self.assertEqual(result.stroke_type, stroke)
                self.assertGreaterEqual(len(result.rule_results), 3)

    def test_rule_score_summary_maps_rule_names_to_scores(self):
        result = evaluate_stroke_rules(_base_sequence(), "clear")
        summary = rule_score_summary(result)
        self.assertTrue(summary)
        self.assertTrue(all(isinstance(v, float) for v in summary.values()))

    def test_unknown_stroke_raises_helpful_error(self):
        with self.assertRaises(ValueError):
            evaluate_stroke_rules(_base_sequence(), "serve")


if __name__ == "__main__":
    unittest.main()
