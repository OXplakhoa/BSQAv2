# Phase 4 — Quality Assessment Implementation Plan

Date: 2026-05-30

## Goal

Implement the Phase 4 quality-assessment layer from `implementation_plan_final.md`:

```text
DTW similarity + biomechanics rule scoring -> hybrid quality score + feedback
```

This is different from the already-completed Observatory pose reliability score.

## What already exists

The Observatory currently has:

- pose reliability score,
- missing joint / visibility / jump diagnostics,
- RF/DL predictions,
- curated diagnostics and error analysis,
- optional DL quality-head field support if checkpoints expose it.

These are useful, but they are **not** a technique-quality scorer.

## What Phase 4 adds

New package:

```text
src/quality/
  dtw_scorer.py
  rules.py
  hybrid.py
src/observatory/
  quality_references.py
```

### 1. DTW scorer

Purpose: compare an input skeleton sequence against reference/pro-style sequences of the same predicted or known stroke type.

Contract:

- accepts `(T, 17, 2)` skeleton arrays,
- optionally focuses on badminton-relevant joints,
- computes normalized Dynamic Time Warping distance,
- converts distance to a 0-100 similarity score,
- returns best reference match metadata.
- Observatory integration loads a small per-stroke reference bank from curated cached PipelineRun artifacts.

### 2. Biomechanics rules

Purpose: provide interpretable stroke-specific checks from skeleton-derived features.

Rules are heuristic and report-friendly, not medical/coaching ground truth.

Initial rule families:

- `smash`: high contact, fast wrist, elbow extension, late impact,
- `clear`: high contact, arm extension, controlled late swing,
- `drop_shot`: high preparation but softer wrist speed/follow-through,
- `net_shot`: low speed, compact motion, low/controlled contact,
- `lift`: upward wrist motion/contact lift, knee/hip involvement, controlled speed.

Each rule returns:

```python
{
  "rule": "contact_height",
  "score": 0..100,
  "feedback": "..."
}
```

### 3. Hybrid scorer

Purpose: combine DTW and rule scores.

Default weights:

```text
DTW:   40%
Rules: 60%
```

Output:

```python
{
  "stroke_type": "smash",
  "quality_score": 0..100,
  "dtw_score": 0..100 or None,
  "rule_score": 0..100,
  "rule_scores": {...},
  "feedback": [...],
  "reference_match": {...}
}
```

## Limitations / honest framing

- This is a heuristic quality scorer because the project has no supervised expert quality labels.
- DTW quality depends heavily on the quality and representativeness of reference samples.
- Rules use 2D pose only; racket, shuttle, court position, and camera angle are missing.
- Scores should be presented as technique-similarity/biomechanical indicators, not certified coaching grades.

## Acceptance criteria

- Identical sequences get near-perfect DTW similarity.
- Degraded/different sequences get lower DTW similarity.
- Each stroke type returns at least three interpretable rule scores.
- Hybrid score combines DTW and rules with configurable weights.
- Missing DTW references still returns a valid rule-only quality report.
- Custom Upload passes curated same-stroke references and displays DTW similarity when available.
- Feedback strings are specific and do not overclaim correctness.
- Tests cover DTW, rules, and hybrid contracts.

## Next after Phase 4

After user review, proceed to Phase 5:

- full evaluation / ablation report generation,
- evaluate.py polishing,
- final report tables and figures.
