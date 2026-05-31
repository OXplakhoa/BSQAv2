import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.observatory.schema import PipelineRun
from src.observatory.upload_pipeline import save_uploaded_file, run_uploaded_video_pipeline


class UploadPipelineTests(unittest.TestCase):
    def test_save_uploaded_file_writes_bytes_with_safe_name(self):
        with tempfile.TemporaryDirectory() as td:
            path = save_uploaded_file(b"video-bytes", Path(td), "../bad name.mp4")
            self.assertTrue(path.exists())
            self.assertEqual(path.read_bytes(), b"video-bytes")
            self.assertTrue(path.name.endswith("bad_name.mp4"))

    def test_run_uploaded_video_pipeline_uses_injected_extractor(self):
        def fake_extractor(path):
            keypoints = np.ones((8, 17, 2), dtype=np.float32)
            visibilities = np.ones((8, 17), dtype=np.float32)
            return keypoints, visibilities, 30

        with tempfile.TemporaryDirectory() as td:
            video_path = Path(td) / "clip.mp4"
            video_path.write_bytes(b"fake")
            run = run_uploaded_video_pipeline(
                video_path,
                sample_id="upload_1",
                ground_truth="smash",
                extractor=fake_extractor,
                run_id="upload_run_1",
            )

        self.assertIsInstance(run, PipelineRun)
        self.assertEqual(run.mode, "upload")
        self.assertEqual(run.sample_id, "upload_1")
        self.assertEqual(run.ground_truth, "smash")
        self.assertEqual(run.video_metadata["fps"], 30)
        self.assertEqual(run.arrays["raw_keypoints"].shape, (8, 17, 2))
        self.assertEqual(run.arrays["normalized_keypoints"].shape, (64, 17, 2))
        self.assertIn("pose_summary", run.diagnostics)


if __name__ == "__main__":
    unittest.main()
