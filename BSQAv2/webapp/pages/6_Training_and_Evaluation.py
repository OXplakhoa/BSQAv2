from components.bootstrap import PROJECT_ROOT, ensure_project_imports
ensure_project_imports()

import pandas as pd
import streamlit as st

from components.dm_viz import existing_path, horizontal_bar_figure, load_json_file
from components.eval_viz import (
    comparison_rows,
    dl_final_metrics,
    fold_artifact_rows,
    metric_card_value,
    per_class_metric_rows,
    rf_summary_metrics,
)
from components.ui import explanation_card, metric_row, render_sidebar


st.set_page_config(page_title="Training & Evaluation", page_icon="📊", layout="wide")

sample_id = render_sidebar()
st.title("Training & Evaluation Dashboard")

explanation_card(
    "How the final models compare",
    "This page separates the academic evaluation story from single-sample demos. "
    "Random Forest is the strongest numerical classifier in the current artifacts, while "
    "GCN + BiLSTM + Attention remains the main Deep Learning architecture because it models "
    "skeleton graphs, temporal motion, and attention-based frame importance.",
)

RESULTS_ROOT = PROJECT_ROOT / "results"
COLAB_RUN_DIR = PROJECT_ROOT / "_colab_results" / "gcn_bilstm_attn_20260528_095136"
RF_RESULTS_PATH = RESULTS_ROOT / "rf_baseline" / "rf_results.json"
TREE_RESULTS_PATH = RESULTS_ROOT / "dm_analysis" / "decision_tree_results.json"
CV_SUMMARY_PATH = COLAB_RUN_DIR / "cv_summary.json"

rf_results = load_json_file(RF_RESULTS_PATH)
tree_results = load_json_file(TREE_RESULTS_PATH)
local_cv_summary = load_json_file(CV_SUMMARY_PATH)
dl_metrics = dl_final_metrics()
rf_metrics = rf_summary_metrics(rf_results)
comparison = comparison_rows(rf_results, tree_results, dl_metrics=dl_metrics)
per_class_rows = per_class_metric_rows(rf_results, dl_metrics=dl_metrics)
checkpoint_paths = sorted(COLAB_RUN_DIR.glob("best_model_fold*.pth")) if COLAB_RUN_DIR.exists() else []
event_paths = sorted(COLAB_RUN_DIR.glob("fold_*/events.out.tfevents.*")) if COLAB_RUN_DIR.exists() else []

st.header("Final headline metrics")
metric_row([
    ("DL accuracy", f"{dl_metrics['accuracy']:.3f} ± {dl_metrics['accuracy_std']:.3f}", "Final GCN + BiLSTM + Attention CV result from report/PRD"),
    ("DL F1 macro", f"{dl_metrics['f1_macro']:.3f} ± {dl_metrics['f1_macro_std']:.3f}", "Macro F1 across five stroke classes"),
    ("RF accuracy", metric_card_value(rf_metrics.get("accuracy")), "Random Forest baseline artifact"),
    ("RF F1 macro", metric_card_value(rf_metrics.get("f1_macro")), "Random Forest macro F1"),
])

st.info(
    "Main conclusion: engineered biomechanical features + Random Forest currently outperform the final DL model numerically. "
    "The DL model is still the proposed spatial-temporal architecture and provides attention-based inspection."
)

st.divider()
st.header("Model comparison")

comparison_df = pd.DataFrame(comparison)
if not comparison_df.empty:
    st.dataframe(comparison_df, width="stretch")
    chart_rows = [
        {"model": row["model"], "accuracy": row["accuracy"]}
        for row in comparison
        if row.get("accuracy") is not None
    ]
    st.pyplot(horizontal_bar_figure(chart_rows, "model", "accuracy", "Accuracy by model"))
else:
    st.warning("No model-comparison artifacts found.")

st.divider()
st.header("Per-class performance")

if per_class_rows:
    per_class_df = pd.DataFrame(per_class_rows)
    st.dataframe(per_class_df, width="stretch")

    dl_chart_rows = [
        {"class": row["class"], "dl_f1": row.get("dl_f1") or 0.0}
        for row in per_class_rows
    ]
    rf_chart_rows = [
        {"class": row["class"], "rf_f1": row.get("rf_f1") or 0.0}
        for row in per_class_rows
    ]
    col_rf, col_dl = st.columns(2)
    with col_rf:
        st.pyplot(horizontal_bar_figure(rf_chart_rows, "class", "rf_f1", "RF per-class F1"))
    with col_dl:
        st.pyplot(horizontal_bar_figure(dl_chart_rows, "class", "dl_f1", "DL per-class F1"))
    st.warning(
        "Lift is the weakest DL class (F1 ≈ 0.434), so lift/clear and lift/net-shot cases are useful error-analysis examples."
    )
else:
    st.info("Per-class metrics are unavailable.")

st.divider()
st.header("Evaluation artifacts")

artifact_cols = st.columns(3)
with artifact_cols[0]:
    st.subheader("RF confusion matrix")
    path = existing_path(RESULTS_ROOT / "rf_baseline" / "rf_confusion_matrix.png")
    if path:
        st.image(str(path), caption="RF confusion matrix")
    else:
        st.info("RF confusion matrix image missing.")

with artifact_cols[1]:
    st.subheader("Normalized RF confusion matrix")
    path = existing_path(RESULTS_ROOT / "rf_baseline" / "rf_confusion_matrix_norm.png")
    if path:
        st.image(str(path), caption="Normalized RF confusion matrix")
    else:
        st.info("Normalized RF confusion matrix image missing.")

with artifact_cols[2]:
    st.subheader("RF feature importance")
    path = existing_path(RESULTS_ROOT / "rf_baseline" / "rf_feature_importance.png")
    if path:
        st.image(str(path), caption="RF feature importance")
    else:
        st.info("RF feature importance image missing.")

st.divider()
st.header("Training run inventory")

fold_rows = fold_artifact_rows(local_cv_summary, [path.name for path in checkpoint_paths]) if local_cv_summary else []
if fold_rows:
    st.dataframe(pd.DataFrame(fold_rows), width="stretch")
else:
    st.info("No local DL CV summary found.")

metric_row([
    ("Checkpoints", str(len(checkpoint_paths)), "best_model_fold*.pth files found locally"),
    ("TensorBoard event files", str(len(event_paths)), "Training event logs found under fold_* directories"),
    ("Local run CV accuracy", metric_card_value(local_cv_summary.get("accuracy_mean") if local_cv_summary else None), "Exploratory local/Colab run summary"),
    ("Local run F1 macro", metric_card_value(local_cv_summary.get("f1_macro_mean") if local_cv_summary else None), "Exploratory local/Colab run summary"),
])

if local_cv_summary:
    st.caption(
        "Note: the checkpoint folder used for demo inference may have exploratory metrics that differ from the final report/PRD values. "
        "The headline DL metrics above use the final selected evaluation values."
    )

with st.expander("Technical: raw metric artifacts", expanded=st.session_state.get("detail_level") == "Technical"):
    st.write("**RF results JSON**")
    st.json(rf_results)
    st.write("**Decision Tree results JSON**")
    st.json(tree_results)
    st.write("**Local DL cv_summary.json**")
    st.json(local_cv_summary)

st.caption(
    "Evaluation dashboard scope: report-ready comparison of RF, Decision Tree, and final GCN + BiLSTM + Attention. "
    "Future work can add training-curve extraction from TensorBoard event files."
)
