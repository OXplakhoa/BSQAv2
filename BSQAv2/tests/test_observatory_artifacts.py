import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from src.observatory.diagnostics import summarize_prediction
from src.observatory.dm_inference import load_rf_bundle
from src.observatory.schema import (
    ArtifactValidationError,
    PipelineRun,
    PredictionResult,
    load_pipeline_run,
    save_pipeline_run,
)


class _FakeLabelEncoder:
    classes_ = np.asarray(["clear", "smash"])

    def inverse_transform(self, values):
        return self.classes_[values]


class _FakeRFModel:
    def predict(self, X):
        return np.asarray([1])

    def predict_proba(self, X):
        return np.asarray([[0.25, 0.75]])


class PipelineRunArtifactTests(unittest.TestCase):
    def test_pipeline_run_round_trip_keeps_arrays_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = PipelineRun.new(
                run_id="test_run",
                sample_id="sample_001",
                mode="curated",
                source_video_path="data/clips/smash/example.mp4",
            )
            run.dl_prediction = PredictionResult.from_probabilities(
                {"clear": 0.2, "smash": 0.8}
            )
            run.arrays["raw_keypoints"] = np.zeros((10, 17, 2), dtype=np.float32)
            run.arrays["normalized_keypoints"] = np.ones((64, 17, 2), dtype=np.float32)

            save_pipeline_run(run, Path(tmp))
            loaded = load_pipeline_run(Path(tmp))

            self.assertEqual(loaded.run_id, "test_run")
            self.assertEqual(loaded.dl_prediction.label, "smash")
            self.assertEqual(loaded.arrays["raw_keypoints"].shape, (10, 17, 2))
            self.assertEqual(loaded.arrays["normalized_keypoints"].shape, (64, 17, 2))

    def test_pipeline_run_rejects_bad_normalized_shape(self):
        run = PipelineRun.new("bad_run", "sample_001", "curated")
        run.arrays["normalized_keypoints"] = np.zeros((63, 17, 2), dtype=np.float32)
        with self.assertRaises(ArtifactValidationError):
            run.validate()


class DiagnosticsTests(unittest.TestCase):
    def test_unknown_ground_truth_does_not_claim_correctness(self):
        prediction = PredictionResult.from_probabilities({"clear": 0.43, "lift": 0.30})
        text = summarize_prediction(prediction, ground_truth=None, pose_reliability=0.6)
        self.assertIn("interpreted cautiously", text)
        self.assertNotIn("correct", text.lower())
        self.assertNotIn("incorrect", text.lower())

    def test_known_ground_truth_reports_correctness(self):
        prediction = PredictionResult.from_probabilities({"clear": 0.8, "lift": 0.2})
        text = summarize_prediction(prediction, ground_truth="clear", pose_reliability=0.9)
        self.assertIn("prediction is correct", text)


class RFBundleTests(unittest.TestCase):
    def test_rf_bundle_vectorizes_in_saved_feature_order_and_imputes_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle_path = Path(tmp) / "rf_model_bundle.joblib"
            joblib.dump(
                {
                    "model": _FakeRFModel(),
                    "label_encoder": _FakeLabelEncoder(),
                    "feature_names": ["f1", "f2"],
                    "feature_medians": {"f1": 1.0, "f2": 2.0},
                    "class_averages": {
                        "clear": {"f1": 0.0, "f2": 2.0},
                        "smash": {"f1": 1.0, "f2": 3.0},
                    },
                    "metadata": {},
                },
                bundle_path,
            )

            bundle = load_rf_bundle(bundle_path)
            X = bundle.vectorize({"f1": np.nan, "f2": 5.0})
            self.assertEqual(X.tolist(), [[1.0, 5.0]])
            prediction = bundle.predict({"f1": np.nan, "f2": 5.0})
            self.assertEqual(prediction.label, "smash")
            self.assertAlmostEqual(prediction.confidence, 0.75)

    def test_rf_bundle_rejects_missing_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle_path = Path(tmp) / "rf_model_bundle.joblib"
            joblib.dump(
                {
                    "model": _FakeRFModel(),
                    "label_encoder": _FakeLabelEncoder(),
                    "feature_names": ["f1", "f2"],
                    "feature_medians": {"f1": 1.0, "f2": 2.0},
                    "class_averages": {"smash": {"f1": 1.0, "f2": 3.0}},
                    "metadata": {},
                },
                bundle_path,
            )

            bundle = load_rf_bundle(bundle_path)
            with self.assertRaises(ArtifactValidationError):
                bundle.vectorize({"f1": 1.0})


if __name__ == "__main__":
    unittest.main()
