import unittest

import matplotlib.pyplot as plt
import numpy as np

from src.observatory.schema import PipelineRun, PredictionResult
from webapp.components.dl_viz import (
    attention_heatmap_figure,
    confidence_interpretation,
    dl_shape_rows,
    top_attention_frames,
)


def _run():
    run = PipelineRun.new("run_1", "sample_1", "curated")
    run.dl_prediction = PredictionResult.from_probabilities({"clear": 0.2, "smash": 0.8})
    attention = np.zeros((64, 64), dtype=np.float32)
    attention[:, 10] = 0.5
    attention[:, 20] = 0.25
    run.arrays["attention_weights"] = attention
    run.diagnostics["dl_shapes"] = {
        "input": [1, 64, 17, 2],
        "logits": [1, 5],
        "attention_weights": [1, 64, 64],
    }
    return run


class DeepLearningVizTests(unittest.TestCase):
    def test_top_attention_frames_are_ranked(self):
        rows = top_attention_frames(_run(), top_n=2)
        self.assertEqual(rows[0]["frame"], 10)
        self.assertEqual(rows[1]["frame"], 20)

    def test_confidence_interpretation_mentions_high_confidence(self):
        text = confidence_interpretation(PredictionResult.from_probabilities({"smash": 0.8, "clear": 0.2}))
        self.assertIn("High DL confidence", text)

    def test_attention_heatmap_returns_figure(self):
        fig = attention_heatmap_figure(_run().arrays["attention_weights"])
        try:
            self.assertEqual(len(fig.axes), 2)  # heatmap + colorbar
        finally:
            plt.close(fig)

    def test_dl_shape_rows_are_extracted(self):
        rows = dl_shape_rows(_run())
        self.assertEqual(rows[0]["tensor"], "input")
        self.assertIn("64", rows[0]["shape"])


if __name__ == "__main__":
    unittest.main()
