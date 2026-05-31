import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import torch

from src.config import STROKE_TYPES
from src.models.gcn_bilstm_attn import GCNBiLSTMAttention
from src.observatory.dl_inference import load_dl_model, run_dl_inference
from src.observatory.pipeline import run_skeleton_pipeline


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


def _dl_checkpoint(path):
    model = GCNBiLSTMAttention()
    prefixed = {f"inner.{key}": value for key, value in model.state_dict().items()}
    torch.save({"epoch": 1, "model_state_dict": prefixed, "val_loss": 1.23}, path)


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


class DLInferenceTests(unittest.TestCase):
    def test_dl_inference_returns_probabilities_and_attention(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "best_model_fold0.pth"
            _dl_checkpoint(checkpoint)

            bundle = load_dl_model(checkpoint, device="cpu")
            result = run_dl_inference(bundle, np.zeros((64, 17, 2), dtype=np.float32))

            self.assertIn(result.prediction.label, STROKE_TYPES)
            self.assertAlmostEqual(sum(result.prediction.probabilities.values()), 1.0, places=5)
            self.assertEqual(result.attention_weights.shape, (64, 64))
            self.assertEqual(result.shape_metadata["input"], [1, 64, 17, 2])

    def test_missing_checkpoint_has_helpful_error(self):
        with self.assertRaisesRegex(FileNotFoundError, "DL checkpoint not found"):
            load_dl_model(Path("missing_checkpoint.pth"), device="cpu")

    def test_skeleton_pipeline_can_include_rf_and_dl_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rf_path = tmp / "rf_model_bundle.joblib"
            dl_path = tmp / "best_model_fold0.pth"
            _rf_bundle(rf_path)
            _dl_checkpoint(dl_path)

            run = run_skeleton_pipeline(
                sample_id="sample_001",
                raw_keypoints=_synthetic_pose(),
                visibilities=np.ones((20, 17)),
                source_video_path="data/clips/smash/example.mp4",
                ground_truth="smash",
                rf_bundle_path=rf_path,
                dl_checkpoint_path=dl_path,
                dl_device="cpu",
            )

            run.validate()
            self.assertIsNotNone(run.dl_prediction.label)
            self.assertEqual(run.rf_prediction.label, "smash")
            self.assertEqual(run.arrays["attention_weights"].shape, (64, 64))
            self.assertIn("dl_summary", run.diagnostics)
            self.assertIn("branch_comparison", run.diagnostics)


if __name__ == "__main__":
    unittest.main()
