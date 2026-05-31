import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.observatory.artifacts import ArtifactRegistry
from src.observatory.quality_references import load_quality_reference_bank, reference_bank_summary
from src.observatory.schema import PipelineRun, save_pipeline_run


class QualityReferenceBankTests(unittest.TestCase):
    def test_loads_curated_normalized_keypoints_by_stroke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "artifacts"
            registry = ArtifactRegistry(root=root)
            registry.ensure_layout()

            run = PipelineRun.new(run_id="run1", sample_id="sample1", mode="skeleton", ground_truth="smash")
            run.pose_qc = {"reliability_score": 0.9}
            run.arrays["normalized_keypoints"] = np.zeros((64, 17, 2), dtype=np.float32)
            save_pipeline_run(run, registry.pipeline_runs_dir / "run1")

            manifest = {
                "samples": [
                    {
                        "sample_id": "sample1",
                        "title": "Sample 1",
                        "stroke_type": "smash",
                        "ground_truth": "smash",
                        "video_path": "data/clips/smash/sample1.mp4",
                        "pipeline_run_dir": "pipeline_runs/run1",
                    }
                ]
            }
            registry.curated_manifest.write_text(json.dumps(manifest), encoding="utf-8")

            bank = load_quality_reference_bank(registry=registry)

            self.assertIn("smash", bank)
            self.assertEqual(bank["smash"][0][0], "sample1")
            self.assertEqual(bank["smash"][0][1].shape, (64, 17, 2))
            self.assertEqual(reference_bank_summary(bank)["smash"], 1)

    def test_filters_low_reliability_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "artifacts"
            registry = ArtifactRegistry(root=root)
            registry.ensure_layout()

            run = PipelineRun.new(run_id="run1", sample_id="sample1", mode="skeleton", ground_truth="clear")
            run.pose_qc = {"reliability_score": 0.1}
            run.arrays["normalized_keypoints"] = np.zeros((64, 17, 2), dtype=np.float32)
            save_pipeline_run(run, registry.pipeline_runs_dir / "run1")

            registry.curated_manifest.write_text(
                json.dumps(
                    {
                        "samples": [
                            {
                                "sample_id": "sample1",
                                "title": "Sample 1",
                                "stroke_type": "clear",
                                "ground_truth": "clear",
                                "video_path": "data/clips/clear/sample1.mp4",
                                "pipeline_run_dir": "pipeline_runs/run1",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            bank = load_quality_reference_bank(registry=registry, min_pose_reliability=0.8)
            self.assertNotIn("clear", bank)


if __name__ == "__main__":
    unittest.main()
