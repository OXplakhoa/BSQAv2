import unittest

from tools.build_curated_manifest import choose_curated_samples


def _row(sample_id, stroke, rf=True, dl=True, pose=0.9, rf_conf=0.7, dl_conf=0.6, warnings=None):
    return {
        "sample_id": sample_id,
        "stroke_type": stroke,
        "video_path": f"data/clips/{stroke}/{sample_id}.mp4",
        "pipeline_run_dir": f"webapp/artifacts/pipeline_runs/{sample_id}",
        "pose_reliability_score": pose,
        "pose_reliability_label": "high" if pose >= 0.8 else "medium",
        "pose_warnings": warnings or [],
        "rf_prediction": stroke if rf else "clear",
        "rf_confidence": rf_conf,
        "rf_correct": rf,
        "dl_prediction": stroke if dl else "net_shot",
        "dl_confidence": dl_conf,
        "dl_correct": dl,
        "rf_dl_agree": rf and dl,
    }


class CuratedManifestBuilderTests(unittest.TestCase):
    def test_manifest_selection_covers_classes_and_teaching_cases(self):
        rows = []
        for stroke in ["smash", "clear", "drop_shot", "net_shot", "lift"]:
            rows.append(_row(f"{stroke}_clean", stroke, rf=True, dl=True, pose=0.95))
            rows.append(_row(f"{stroke}_warn", stroke, rf=True, dl=False, pose=0.65, warnings=["low wrist"]))
            rows.append(_row(f"{stroke}_lowconf", stroke, rf=True, dl=False, pose=0.85, rf_conf=0.35))

        samples = choose_curated_samples(rows, target_count=12)

        self.assertEqual(len(samples), 12)
        self.assertEqual(len({sample["sample_id"] for sample in samples}), 12)
        covered = {sample["stroke_type"] for sample in samples}
        self.assertEqual(covered, {"smash", "clear", "drop_shot", "net_shot", "lift"})

        tags = {tag for sample in samples for tag in sample["tags"]}
        self.assertIn("rf_correct_dl_wrong", tags)
        self.assertIn("pose_warning", tags)
        self.assertIn("low_rf_confidence", tags)

    def test_manifest_entries_include_required_metadata(self):
        rows = [_row(f"{stroke}_clean", stroke) for stroke in ["smash", "clear", "drop_shot", "net_shot", "lift"]]
        samples = choose_curated_samples(rows, target_count=5)

        required = {
            "sample_id",
            "title",
            "stroke_type",
            "ground_truth",
            "video_path",
            "pipeline_run_dir",
            "manual_review_status",
            "teaching_point",
            "diagnosis",
            "tags",
        }
        for sample in samples:
            self.assertTrue(required.issubset(sample.keys()))
            self.assertEqual(sample["manual_review_status"], "reviewed")
            self.assertEqual(sample["ground_truth"], sample["stroke_type"])


if __name__ == "__main__":
    unittest.main()
