"""Dynamic Time Warping quality scorer for skeleton sequences.

This module compares a query stroke against one or more reference strokes of the
same class.  Scores are heuristic 0-100 similarity indicators, not supervised
expert-quality labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_DTW_JOINTS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]


@dataclass
class DTWMatchResult:
    best_reference_id: Optional[str]
    distance: Optional[float]
    score: Optional[float]
    n_references: int


def _prepare_sequence(sequence: np.ndarray, joints: Optional[Sequence[int]] = None) -> np.ndarray:
    arr = np.asarray(sequence, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1:] != (17, 2):
        raise ValueError(f"sequence must have shape (T, 17, 2); got {arr.shape}")
    if joints is not None:
        arr = arr[:, list(joints), :]
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr.reshape(arr.shape[0], -1)


def dtw_distance(
    query: np.ndarray,
    reference: np.ndarray,
    joints: Optional[Sequence[int]] = DEFAULT_DTW_JOINTS,
) -> float:
    """Return normalized DTW distance between two skeleton sequences."""
    q = _prepare_sequence(query, joints=joints)
    r = _prepare_sequence(reference, joints=joints)
    if q.shape[0] == 0 or r.shape[0] == 0:
        raise ValueError("DTW requires non-empty sequences")

    n, m = q.shape[0], r.shape[0]
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    steps = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = float(np.linalg.norm(q[i - 1] - r[j - 1]) / max(1, q.shape[1]))
            choices = [
                (dp[i - 1, j], steps[i - 1, j]),
                (dp[i, j - 1], steps[i, j - 1]),
                (dp[i - 1, j - 1], steps[i - 1, j - 1]),
            ]
            prev_cost, prev_steps = min(choices, key=lambda item: item[0])
            dp[i, j] = cost + prev_cost
            steps[i, j] = prev_steps + 1

    path_len = max(1, int(steps[n, m]))
    return float(dp[n, m] / path_len)


def dtw_similarity_score(distance: float, scale: float = 0.35) -> float:
    """Convert normalized DTW distance into a bounded 0-100 similarity score."""
    distance = max(0.0, float(distance))
    scale = max(1e-6, float(scale))
    return float(np.clip(100.0 * np.exp(-distance / scale), 0.0, 100.0))


def score_against_references(
    query: np.ndarray,
    references: Iterable[Tuple[str, np.ndarray]],
    joints: Optional[Sequence[int]] = DEFAULT_DTW_JOINTS,
    scale: float = 0.35,
) -> DTWMatchResult:
    """Score query against references and return the best DTW match."""
    refs = list(references)
    if not refs:
        return DTWMatchResult(best_reference_id=None, distance=None, score=None, n_references=0)

    best_id: Optional[str] = None
    best_distance: Optional[float] = None
    for ref_id, ref_seq in refs:
        distance = dtw_distance(query, ref_seq, joints=joints)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_id = str(ref_id)

    assert best_distance is not None
    return DTWMatchResult(
        best_reference_id=best_id,
        distance=float(best_distance),
        score=dtw_similarity_score(best_distance, scale=scale),
        n_references=len(refs),
    )
