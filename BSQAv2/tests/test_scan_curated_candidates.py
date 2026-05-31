import unittest

from tools.scan_curated_candidates import select_candidate_videos


class CandidateScanTests(unittest.TestCase):
    def test_candidate_selection_is_class_balanced_and_deterministic(self):
        candidates = []
        for stroke in ["smash", "clear", "drop_shot", "net_shot", "lift"]:
            for idx in range(5):
                candidates.append(
                    {
                        "sample_id": f"{stroke}_{idx}",
                        "stroke_type": stroke,
                        "video_path": f"data/clips/{stroke}/{idx}.mp4",
                    }
                )

        first = select_candidate_videos(candidates, max_total=10, limit_per_class=2, seed=42)
        second = select_candidate_videos(candidates, max_total=10, limit_per_class=2, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)
        counts = {}
        for item in first:
            counts[item["stroke_type"]] = counts.get(item["stroke_type"], 0) + 1
        self.assertEqual(counts, {
            "smash": 2,
            "clear": 2,
            "drop_shot": 2,
            "net_shot": 2,
            "lift": 2,
        })


if __name__ == "__main__":
    unittest.main()
