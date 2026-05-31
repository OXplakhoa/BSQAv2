"""Hybrid DTW + biomechanics-rule quality scorer."""
from __future__ import annotations

from dataclasses import asdict
from typing import Iterable, Optional, Tuple

import numpy as np

from .dtw_scorer import DTWMatchResult, score_against_references
from .rules import evaluate_stroke_rules, rule_score_summary


class HybridQualityScorer:
    """Combine reference similarity and rule-based biomechanics scoring.

    Default weighting follows the original Phase 4 plan:

    - DTW similarity: 40%
    - biomechanics rules: 60%

    If no references are supplied, the scorer returns a valid rule-only report.
    """

    def __init__(self, dtw_weight: float = 0.4, rule_weight: float = 0.6):
        total = float(dtw_weight) + float(rule_weight)
        if total <= 0:
            raise ValueError("dtw_weight + rule_weight must be > 0")
        self.dtw_weight = float(dtw_weight) / total
        self.rule_weight = float(rule_weight) / total

    def score(
        self,
        keypoints: np.ndarray,
        stroke_type: str,
        references: Optional[Iterable[Tuple[str, np.ndarray]]] = None,
    ) -> dict:
        rule_eval = evaluate_stroke_rules(keypoints, stroke_type)
        rule_score = float(rule_eval.overall_score)

        dtw_result: DTWMatchResult = score_against_references(keypoints, references or [])
        if dtw_result.score is None:
            quality_score = rule_score
        else:
            quality_score = (self.dtw_weight * float(dtw_result.score)) + (self.rule_weight * rule_score)

        quality_score = float(np.clip(quality_score, 0.0, 100.0))
        return {
            "stroke_type": stroke_type,
            "quality_score": quality_score,
            "dtw_score": dtw_result.score,
            "rule_score": rule_score,
            "rule_scores": rule_score_summary(rule_eval),
            "feedback": list(rule_eval.feedback),
            "reference_match": asdict(dtw_result),
            "weights": {
                "dtw": self.dtw_weight if dtw_result.score is not None else 0.0,
                "rules": self.rule_weight if dtw_result.score is not None else 1.0,
            },
        }
