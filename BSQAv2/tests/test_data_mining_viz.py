import json
import tempfile
import unittest
from pathlib import Path

from webapp.components.dm_viz import (
    class_average_comparison_rows,
    entropy_class_rows,
    feature_value_rows,
    load_json_file,
    mutual_information_rows,
    rf_top_feature_rows,
)


class _FakeBundle:
    feature_names = ["speed", "angle", "contact_height"]

    def compare_to_class_average(self, features, class_name, top_n=10):
        return [
            {
                "feature": "speed",
                "value": features["speed"],
                "class_average": 3.0,
                "absolute_delta": abs(features["speed"] - 3.0),
            },
            {
                "feature": "angle",
                "value": features["angle"],
                "class_average": 45.0,
                "absolute_delta": abs(features["angle"] - 45.0),
            },
        ][:top_n]


class DataMiningVizTests(unittest.TestCase):
    def test_feature_value_rows_follow_feature_order_and_format_values(self):
        rows = feature_value_rows(
            {"angle": 45.12345, "speed": 3.2, "unused": 99},
            feature_order=["speed", "angle"],
        )
        self.assertEqual([row["feature"] for row in rows], ["speed", "angle", "unused"])
        self.assertEqual(rows[0]["value"], 3.2)
        self.assertEqual(rows[0]["value_display"], "3.2000")

    def test_rf_top_feature_rows_join_current_sample_values(self):
        rf_results = {
            "top_features": [
                {"rank": 1, "feature": "speed", "importance": 0.7},
                {"rank": 2, "feature": "missing", "importance": 0.2},
                {"rank": 3, "feature": "angle", "importance": 0.1},
            ]
        }
        rows = rf_top_feature_rows(rf_results, {"speed": 3.2, "angle": 45.0}, top_n=5)
        self.assertEqual([row["feature"] for row in rows], ["speed", "missing", "angle"])
        self.assertEqual(rows[0]["current_value"], 3.2)
        self.assertIsNone(rows[1]["current_value"])

    def test_entropy_class_rows_parse_distribution(self):
        rows = entropy_class_rows({
            "class_distribution": {
                "smash": {"count": 8, "prob": 0.8, "entropy_contribution": 0.2575},
                "lift": {"count": 2, "prob": 0.2, "entropy_contribution": 0.4644},
            }
        })
        self.assertEqual(rows[0]["class"], "smash")
        self.assertEqual(rows[0]["count"], 8)
        self.assertAlmostEqual(rows[1]["probability"], 0.2)

    def test_mutual_information_rows_read_csv_and_limit(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "mi.csv"
            path.write_text(
                "feature,mutual_information,rank,normalized\n"
                "speed,0.5,1,1.0\n"
                "angle,0.25,2,0.5\n",
                encoding="utf-8",
            )
            rows = mutual_information_rows(path, top_n=1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["feature"], "speed")
        self.assertEqual(rows[0]["rank"], 1)

    def test_class_average_comparison_rows_delegate_to_bundle(self):
        rows = class_average_comparison_rows(_FakeBundle(), {"speed": 4.5, "angle": 40.0}, "smash", top_n=2)
        self.assertEqual(rows[0]["feature"], "speed")
        self.assertEqual(rows[0]["class_average"], 3.0)
        self.assertEqual(rows[1]["absolute_delta"], 5.0)

    def test_load_json_file_returns_empty_dict_for_missing_file(self):
        self.assertEqual(load_json_file(Path("does-not-exist.json")), {})
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "payload.json"
            path.write_text(json.dumps({"ok": True}), encoding="utf-8")
            self.assertEqual(load_json_file(path), {"ok": True})


if __name__ == "__main__":
    unittest.main()
