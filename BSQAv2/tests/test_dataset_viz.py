import tempfile
import unittest
from pathlib import Path

from src.observatory.schema import CuratedSample
from webapp.components.dataset_viz import (
    class_distribution_rows,
    curated_sample_rows,
    dataset_csv_summaries,
    pose_reliability_rows,
    source_summary_rows,
)


class DatasetVizTests(unittest.TestCase):
    def test_dataset_csv_summaries_count_rows_unique_ids_and_classes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "youtube"
            source.mkdir()
            (source / "smash_v2.csv").write_text(
                "id,type_of_shot,frame_count,kpt_0_x,kpt_0_y\n"
                "1,smash,0,0,0\n"
                "1,smash,1,0,0\n"
                "2,smash,0,0,0\n",
                encoding="utf-8",
            )
            rows = dataset_csv_summaries(root, sources=["youtube"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["rows"], 3)
        self.assertEqual(rows[0]["samples"], 2)
        self.assertEqual(rows[0]["stroke_type"], "smash")

    def test_class_distribution_rows_aggregate_samples_by_stroke(self):
        rows = class_distribution_rows([
            {"stroke_type": "smash", "samples": 2, "rows": 10},
            {"stroke_type": "smash", "samples": 3, "rows": 20},
            {"stroke_type": "lift", "samples": 1, "rows": 5},
        ])
        self.assertEqual(rows[0]["stroke_type"], "smash")
        self.assertEqual(rows[0]["samples"], 5)
        self.assertEqual(rows[0]["rows"], 30)

    def test_source_summary_rows_aggregate_by_source(self):
        rows = source_summary_rows([
            {"source": "youtube", "samples": 2, "rows": 10},
            {"source": "youtube", "samples": 3, "rows": 20},
            {"source": "arxiv", "samples": 1, "rows": 5},
        ])
        youtube = [row for row in rows if row["source"] == "youtube"][0]
        self.assertEqual(youtube["files"], 2)
        self.assertEqual(youtube["samples"], 5)

    def test_curated_sample_rows_expose_tags_and_selection(self):
        samples = [
            CuratedSample("s1", "Title 1", "smash", "v1.mp4", tags=["clean"]),
            CuratedSample("s2", "Title 2", "lift", "v2.mp4", tags=["pose_warning"]),
        ]
        rows = curated_sample_rows(samples, selected_sample_id="s2")
        self.assertFalse(rows[0]["selected"])
        self.assertTrue(rows[1]["selected"])
        self.assertEqual(rows[1]["tags"], "pose_warning")

    def test_pose_reliability_rows_use_manifest_pose_metadata(self):
        payload = {
            "samples": [
                {"sample_id": "s1", "stroke_type": "smash", "pose_reliability_score": 0.9, "pose_reliability_label": "high", "pose_warnings": []},
                {"sample_id": "s2", "stroke_type": "lift", "pose_reliability_score": 0.6, "pose_reliability_label": "medium", "pose_warnings": ["wrist"]},
            ]
        }
        rows = pose_reliability_rows(payload)
        self.assertEqual(rows[0]["sample_id"], "s1")
        self.assertEqual(rows[1]["warning_count"], 1)


if __name__ == "__main__":
    unittest.main()
