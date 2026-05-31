import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from src.data.preprocessing import preprocess_sequence
from src.observatory.pipeline import run_skeleton_pipeline
from src.observatory.pose_quality import evaluate_pose_quality


class _FakeLabelEncoder:
    classes_ = np.asarray(["clear", "smash"])

    def inverse_transform(self, values):
        return self.classes_[values]


class _FakeRFModel:
    def predict(self, X):
        return np.asarray([1])

    def predict_proba(self, X):
        return np.asarray([[0.2, 0.8]])


def _synthetic_pose(frames=20):
    keypoints = np.zeros((frames, 17, 2), dtype=np.float32)
    base = np.asarray(
        [
            [320, 120], [305, 110], [335, 110], [295, 118], [345, 118],
            [280, 200], [360, 200], [260, 280], [380, 280], [245, 360],
            [395, 360], [290, 380], [350, 380], [285, 500], [355, 500],
            [280, 620], [360, 620],
        ],
        dtype=np.float32,
    )
    for t in range(frames):
        keypoints[t] = base + np.asarray([t * 2, np.sin(t / 3) * 3], dtype=np.float32)
    return keypoints


def _rf_bundle(path):
    joblib.dump(
        {
            "model": _FakeRFModel(),
            "label_encoder": _FakeLabelEncoder(),
            "feature_names": ["num_frames", "impact_frame", "swing_phase_ratio"],
            "feature_medians": {
                "num_frames": 64.0,
                "impact_frame": 0.5,
                "swing_phase_ratio": 0.5,
            },
            "class_averages": {
                "clear": {"num_frames": 64.0, "impact_frame": 0.3, "swing_phase_ratio": 0.3},
                "smash": {"num_frames": 64.0, "impact_frame": 0.7, "swing_phase_ratio": 0.7},
            },
            "metadata": {},
        },
        path,
    )


class PoseQualityTests(unittest.TestCase):
    def test_good_pose_is_high_reliability(self):
        qc = evaluate_pose_quality(_synthetic_pose(), visibilities=np.ones((20, 17)))

        self.assertGreaterEqual(qc["reliability_score"], 0.8)
        self.assertEqual(qc["reliability_label"], "high")
        self.assertEqual(qc["warnings"], [])

    def test_missing_critical_wrist_lowers_reliability_and_warns(self):
        keypoints = _synthetic_pose()
        visibilities = np.ones((20, 17), dtype=np.float32)
        keypoints[:, 10] = 0.0
        visibilities[:, 10] = 0.0

        qc = evaluate_pose_quality(keypoints, visibilities=visibilities)

        self.assertLess(qc["reliability_score"], 0.8)
        self.assertTrue(any("right_wrist" in warning for warning in qc["warnings"]))

    def test_large_joint_jump_is_reported(self):
        keypoints = _synthetic_pose()
        keypoints[10, 9] += np.asarray([500.0, 500.0], dtype=np.float32)

        qc = evaluate_pose_quality(keypoints, visibilities=np.ones((20, 17)))

        self.assertGreater(qc["outlier_jump_count"], 0)
        self.assertTrue(any("jump" in warning.lower() for warning in qc["warnings"]))


class SkeletonPipelineTests(unittest.TestCase):
    def test_skeleton_pipeline_returns_valid_pipeline_run_with_rf_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle_path = Path(tmp) / "rf_model_bundle.joblib"
            _rf_bundle(bundle_path)

            run = run_skeleton_pipeline(
                sample_id="sample_001",
                raw_keypoints=_synthetic_pose(),
                visibilities=np.ones((20, 17)),
                source_video_path="data/clips/smash/example.mp4",
                ground_truth=None,
                rf_bundle_path=bundle_path,
                quality_references={"smash": [("same_pose", preprocess_sequence(_synthetic_pose()))]},
            )

            run.validate()
            self.assertEqual(run.sample_id, "sample_001")
            self.assertEqual(run.mode, "skeleton")
            self.assertEqual(run.arrays["raw_keypoints"].shape, (20, 17, 2))
            self.assertEqual(run.arrays["normalized_keypoints"].shape, (64, 17, 2))
            self.assertEqual(run.rf_prediction.label, "smash")
            self.assertAlmostEqual(run.rf_prediction.confidence, 0.8)
            self.assertIn("interpreted cautiously", run.diagnostics["rf_summary"])
            self.assertIn("quality_report", run.diagnostics)
            self.assertEqual(run.diagnostics["quality_report"]["stroke_type"], "smash")
            self.assertGreaterEqual(run.diagnostics["quality_report"]["quality_score"], 0)
            self.assertAlmostEqual(run.diagnostics["quality_report"]["dtw_score"], 100.0)
            self.assertEqual(run.diagnostics["quality_report"]["reference_match"]["best_reference_id"], "same_pose")


if __name__ == "__main__":
    unittest.main()
