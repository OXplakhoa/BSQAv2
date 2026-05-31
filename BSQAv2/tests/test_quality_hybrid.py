import unittest

import numpy as np

from src.quality.hybrid import HybridQualityScorer


class HybridQualityScorerTests(unittest.TestCase):
    def test_rule_only_report_when_no_references(self):
        seq = np.zeros((64, 17, 2), dtype=np.float32)
        scorer = HybridQualityScorer()
        report = scorer.score(seq, "smash", references=[])
        self.assertEqual(report["stroke_type"], "smash")
        self.assertIsNone(report["dtw_score"])
        self.assertGreaterEqual(report["quality_score"], 0)
        self.assertTrue(report["feedback"])

    def test_hybrid_report_includes_dtw_when_references_exist(self):
        seq = np.zeros((64, 17, 2), dtype=np.float32)
        scorer = HybridQualityScorer(dtw_weight=0.4, rule_weight=0.6)
        report = scorer.score(seq, "net_shot", references=[("same", seq.copy())])
        self.assertEqual(report["stroke_type"], "net_shot")
        self.assertAlmostEqual(report["dtw_score"], 100.0)
        self.assertIn("rule_score", report)
        self.assertIn("reference_match", report)
        self.assertEqual(report["reference_match"]["best_reference_id"], "same")

    def test_weights_are_normalized(self):
        scorer = HybridQualityScorer(dtw_weight=2.0, rule_weight=1.0)
        self.assertAlmostEqual(scorer.dtw_weight + scorer.rule_weight, 1.0)


if __name__ == "__main__":
    unittest.main()
