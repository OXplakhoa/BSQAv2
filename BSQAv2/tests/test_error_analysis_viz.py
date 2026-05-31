import unittest

from src.observatory.schema import CuratedSample, PipelineRun, PredictionResult
from webapp.components.error_viz import (
    branch_agreement_label,
    curated_error_case_rows,
    known_confusion_note,
    pose_risk_rows,
    prediction_audit_rows,
    probability_comparison_rows,
)


def _run():
    run = PipelineRun.new("run_1", "sample_1", "curated", ground_truth="lift")
    run.rf_prediction = PredictionResult.from_probabilities({"lift": 0.60, "clear": 0.25, "smash": 0.15})
    run.dl_prediction = PredictionResult.from_probabilities({"lift": 0.20, "clear": 0.70, "smash": 0.10})
    run.pose_qc = {
        "reliability_score": 0.62,
        "reliability_label": "medium",
        "warnings": ["low wrist visibility", "large joint jump"],
    }
    run.diagnostics = {"branch_comparison": "The branches disagree."}
    return run


class ErrorAnalysisVizTests(unittest.TestCase):
    def test_prediction_audit_rows_include_correctness_and_margin(self):
        rows = prediction_audit_rows(_run(), ground_truth="lift")
        self.assertEqual(rows[0]["branch"], "Random Forest")
        self.assertEqual(rows[0]["status"], "correct")
        self.assertAlmostEqual(rows[0]["margin"], 0.35)
        self.assertEqual(rows[1]["status"], "incorrect")

    def test_prediction_audit_rows_do_not_claim_correctness_without_ground_truth(self):
        rows = prediction_audit_rows(_run(), ground_truth=None)
        self.assertEqual(rows[0]["status"], "unknown")
        self.assertEqual(rows[1]["status"], "unknown")

    def test_probability_comparison_rows_merge_rf_and_dl_classes(self):
        rows = probability_comparison_rows(_run())
        self.assertEqual(rows[0]["class"], "clear")
        self.assertAlmostEqual(rows[0]["absolute_delta"], 0.45)
        self.assertIn("rf_probability", rows[0])
        self.assertIn("dl_probability", rows[0])

    def test_branch_agreement_label_identifies_disagreement(self):
        self.assertEqual(branch_agreement_label(_run()), "disagree")
        run = _run()
        run.dl_prediction = PredictionResult.from_probabilities({"lift": 0.8, "clear": 0.2})
        self.assertEqual(branch_agreement_label(run), "agree")

    def test_pose_risk_rows_extract_score_label_and_warnings(self):
        rows = pose_risk_rows(_run())
        self.assertEqual(rows[0]["factor"], "Pose reliability")
        self.assertEqual(rows[0]["level"], "medium")
        self.assertEqual(rows[1]["detail"], "low wrist visibility")

    def test_known_confusion_note_explains_lift_clear(self):
        note = known_confusion_note("lift", "clear")
        self.assertIn("lift and clear", note.lower())
        self.assertEqual(known_confusion_note("smash", "smash"), "Predictions agree, so no class-confusion note is needed.")

    def test_curated_error_case_rows_filter_teaching_cases(self):
        samples = [
            CuratedSample("s1", "Clean", "smash", "v1.mp4", tags=["rf_correct"]),
            CuratedSample("s2", "DL wrong", "lift", "v2.mp4", tags=["rf_dl_disagreement"]),
            CuratedSample("s3", "Pose warning", "clear", "v3.mp4", tags=["pose_warning"]),
        ]
        rows = curated_error_case_rows(samples, selected_sample_id="s3")
        self.assertEqual([row["sample_id"] for row in rows], ["s2", "s3"])
        self.assertTrue(rows[1]["selected"])


if __name__ == "__main__":
    unittest.main()
