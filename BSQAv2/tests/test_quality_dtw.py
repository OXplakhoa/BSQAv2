import unittest

import numpy as np

from src.quality.dtw_scorer import dtw_distance, dtw_similarity_score, score_against_references


class DTWScorerTests(unittest.TestCase):
    def test_identical_sequences_have_zero_distance_and_perfect_similarity(self):
        seq = np.zeros((4, 17, 2), dtype=np.float32)
        seq[:, 9, 0] = [0, 1, 2, 3]
        distance = dtw_distance(seq, seq)
        self.assertAlmostEqual(distance, 0.0)
        self.assertAlmostEqual(dtw_similarity_score(distance), 100.0)

    def test_shifted_sequence_has_lower_similarity(self):
        a = np.zeros((4, 17, 2), dtype=np.float32)
        b = np.ones((4, 17, 2), dtype=np.float32)
        distance = dtw_distance(a, b)
        self.assertGreater(distance, 0.0)
        self.assertLess(dtw_similarity_score(distance), 100.0)

    def test_score_against_references_returns_best_match(self):
        query = np.zeros((4, 17, 2), dtype=np.float32)
        good = query.copy()
        bad = np.ones((4, 17, 2), dtype=np.float32)
        result = score_against_references(query, [("bad", bad), ("good", good)])
        self.assertEqual(result.best_reference_id, "good")
        self.assertAlmostEqual(result.score, 100.0)

    def test_empty_references_return_missing_result(self):
        result = score_against_references(np.zeros((4, 17, 2), dtype=np.float32), [])
        self.assertIsNone(result.best_reference_id)
        self.assertIsNone(result.distance)
        self.assertIsNone(result.score)


if __name__ == "__main__":
    unittest.main()
