"""Report-ready ablation metrics for trained BSQAv2 DL variants.

These values are consolidated from the best available 5-fold ``runs/*/cv_summary.json``
artifacts in this workspace.  Quality MAE / Spearman are intentionally ``None``
for these classification-only training runs because the project has no expert
quality-score labels and the Phase 3 trainer optimized only stroke-class CE loss.
"""
from __future__ import annotations

from typing import Any, Dict, List


ABLATION_MODEL_ROWS: List[Dict[str, Any]] = [
    {
        "model": "LSTM",
        "family": "Deep Learning ablation",
        "accuracy": 0.7224640383683684,
        "accuracy_std": 0.025531354026031763,
        "f1_macro": 0.7128136595032156,
        "f1_macro_std": 0.028688759460975287,
        "f1_weighted": 0.7221413834326517,
        "f1_weighted_std": 0.02516828197879917,
        "quality_mae": "N/A",
        "quality_spearman_rs": "N/A",
        "inference_ms_per_frame": 0.0123,
        "source_run": "runs/lstm_baseline_20260527_105124/cv_summary.json",
        "note": "Best available LSTM baseline artifact; classification-only run.",
    },
    {
        "model": "BiLSTM",
        "family": "Deep Learning ablation",
        "accuracy": 0.7050497903472601,
        "accuracy_std": 0.02873671002087577,
        "f1_macro": 0.699015118743667,
        "f1_macro_std": 0.030681939052584494,
        "f1_weighted": 0.7007323271176606,
        "f1_weighted_std": 0.03177789740877612,
        "quality_mae": "N/A",
        "quality_spearman_rs": "N/A",
        "inference_ms_per_frame": 0.0290,
        "source_run": "runs/bilstm_baseline_20260527_095917/cv_summary.json",
        "note": "Best available BiLSTM baseline artifact; classification-only run.",
    },
    {
        "model": "GCN + LSTM",
        "family": "Deep Learning ablation",
        "accuracy": 0.4508058608058608,
        "accuracy_std": 0.07882308971627623,
        "f1_macro": 0.3864992561231979,
        "f1_macro_std": 0.1082458010013253,
        "f1_weighted": 0.4029144108711263,
        "f1_weighted_std": 0.09351528438286265,
        "quality_mae": "N/A",
        "quality_spearman_rs": "N/A",
        "inference_ms_per_frame": 0.0361,
        "source_run": "runs/gcn_lstm_20260526_205033/cv_summary.json",
        "note": "Best available GCN+LSTM ablation artifact; classification-only run.",
    },
]

QUALITY_METRIC_NOTE = (
    "N/A: these runs were trained for stroke classification only. The dataset does "
    "not contain expert quality-score labels, so MAE and Spearman rs cannot be "
    "computed honestly for the ablation models. Phase 4 quality scoring is a "
    "separate heuristic DTW+rules layer."
)
