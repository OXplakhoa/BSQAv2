"""Random Forest inference helpers for the Pipeline Observatory."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np

from .schema import ArtifactValidationError, PredictionResult


@dataclass
class RFFeatureBundle:
    """Loadable Random Forest artifact with feature metadata."""

    model: object
    label_encoder: object
    feature_names: List[str]
    feature_medians: Dict[str, float]
    class_averages: Dict[str, Dict[str, float]]
    metadata: Dict

    @classmethod
    def load(cls, path: Path) -> "RFFeatureBundle":
        path = Path(path)
        if not path.exists():
            raise ArtifactValidationError(f"RF bundle not found: {path}")
        payload = joblib.load(path)
        required = [
            "model",
            "label_encoder",
            "feature_names",
            "feature_medians",
            "class_averages",
            "metadata",
        ]
        missing = [name for name in required if name not in payload]
        if missing:
            raise ArtifactValidationError(
                f"RF bundle missing required fields: {', '.join(missing)}"
            )
        return cls(**{name: payload[name] for name in required})

    def vectorize(self, features: Dict[str, float]) -> np.ndarray:
        """Return a single-row feature matrix in training feature order."""
        missing = [name for name in self.feature_names if name not in features]
        if missing:
            raise ArtifactValidationError(
                "DM features do not match RF artifact. Missing features: "
                f"{', '.join(missing[:10])}"
            )

        values = []
        for name in self.feature_names:
            value = features.get(name)
            if value is None or not np.isfinite(value):
                value = self.feature_medians.get(name, 0.0)
            values.append(float(value))
        return np.asarray([values], dtype=np.float32)

    def predict(self, features: Dict[str, float]) -> PredictionResult:
        X = self.vectorize(features)
        pred_index = int(self.model.predict(X)[0])
        labels = [str(label) for label in self.label_encoder.classes_]

        probabilities: Dict[str, float] = {}
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            probabilities = {
                label: float(proba[idx]) for idx, label in enumerate(labels)
            }

        if probabilities:
            return PredictionResult.from_probabilities(
                probabilities,
                predicted_index=pred_index,
            )

        label = str(self.label_encoder.inverse_transform([pred_index])[0])
        return PredictionResult(label=label, confidence=None, predicted_index=pred_index)

    def compare_to_class_average(
        self,
        features: Dict[str, float],
        class_name: str,
        top_n: int = 10,
    ) -> List[Dict[str, float]]:
        """Rank features by absolute difference from one class average."""
        averages = self.class_averages.get(class_name)
        if averages is None:
            raise ArtifactValidationError(f"No class averages for class: {class_name}")

        rows = []
        for name in self.feature_names:
            value = features.get(name)
            average = averages.get(name)
            if value is None or average is None:
                continue
            if not np.isfinite(value) or not np.isfinite(average):
                continue
            rows.append({
                "feature": name,
                "value": float(value),
                "class_average": float(average),
                "absolute_delta": float(abs(value - average)),
            })
        rows.sort(key=lambda item: item["absolute_delta"], reverse=True)
        return rows[:top_n]


def load_rf_bundle(path: Path) -> RFFeatureBundle:
    return RFFeatureBundle.load(path)
