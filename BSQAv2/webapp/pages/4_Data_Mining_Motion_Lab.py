from components.bootstrap import PROJECT_ROOT, ensure_project_imports
ensure_project_imports()

import pandas as pd
import streamlit as st

from components.charts import probability_rows
from components.data import load_selected_case
from components.dm_viz import (
    class_average_comparison_rows,
    decision_tree_split_rows,
    entropy_class_rows,
    entropy_information_gain_rows,
    existing_path,
    feature_value_rows,
    horizontal_bar_figure,
    load_json_file,
    load_text_file,
    mutual_information_rows,
    rf_top_feature_rows,
)
from components.ui import explanation_card, metric_row, render_sidebar, show_video
from src.observatory.artifacts import ArtifactRegistry
from src.observatory.dm_inference import load_rf_bundle
from src.observatory.schema import ArtifactValidationError


st.set_page_config(page_title="Data Mining Motion Lab", page_icon="⛏️", layout="wide")

sample_id = render_sidebar()
st.title("Data Mining Motion Lab")

explanation_card(
    "Engineered motion features + interpretable mining",
    "This page connects the selected clip to the Data Mining branch. The RF model uses "
    "biomechanical features extracted from the skeleton sequence; the global figures show "
    "which features reduce class uncertainty across the dataset, while the local tables show "
    "what values were measured for this specific stroke.",
)

if sample_id is None:
    st.stop()

case = load_selected_case(sample_id)
sample = case.sample
run = case.run

RESULTS_ROOT = PROJECT_ROOT / "results"
RF_RESULTS_PATH = RESULTS_ROOT / "rf_baseline" / "rf_results.json"
DM_RESULTS_ROOT = RESULTS_ROOT / "dm_analysis"
ENTROPY_PATH = DM_RESULTS_ROOT / "entropy_analysis.json"
MI_CSV_PATH = DM_RESULTS_ROOT / "mutual_information.csv"
TREE_RESULTS_PATH = DM_RESULTS_ROOT / "decision_tree_results.json"
TREE_RULES_PATH = DM_RESULTS_ROOT / "decision_tree_rules.txt"

rf_results = load_json_file(RF_RESULTS_PATH)
entropy_payload = load_json_file(ENTROPY_PATH)
tree_results = load_json_file(TREE_RESULTS_PATH)
tree_rules = load_text_file(TREE_RULES_PATH)
mi_rows = mutual_information_rows(MI_CSV_PATH, top_n=15)

registry = ArtifactRegistry()
rf_bundle = None
rf_bundle_error = ""
try:
    rf_bundle = load_rf_bundle(registry.resolve_rf_bundle())
except ArtifactValidationError as exc:
    rf_bundle_error = str(exc)

left, right = st.columns([1.05, 1])
with left:
    st.subheader(sample.title)
    show_video(sample.video_path)
with right:
    st.subheader("Local RF prediction")
    metric_row([
        ("Ground truth", sample.ground_truth or "unknown", "Manual curated label"),
        ("RF prediction", run.rf_prediction.label or "missing", "Random Forest on biomechanical features"),
        ("RF confidence", f"{(run.rf_prediction.confidence or 0):.3f}", "Top RF class probability"),
        ("Feature count", str(len(run.dm_features)), "Biomechanical features extracted from skeleton"),
    ])
    prob_rows = probability_rows(run.rf_prediction.probabilities)
    if prob_rows:
        st.pyplot(horizontal_bar_figure(
            prob_rows,
            label_key="class",
            value_key="probability",
            title="RF class probabilities",
        ))
    else:
        st.info("No RF probability data available for this PipelineRun.")
    st.write(run.diagnostics.get("rf_summary", "No RF summary available."))

st.divider()
st.header("Global dataset mining")

entropy_cols = st.columns(4)
entropy_cols[0].metric(
    "Class entropy H(Y)",
    f"{entropy_payload.get('entropy_H_Y_bits', 0):.3f} bits" if entropy_payload else "missing",
    help="How uncertain the stroke class is before seeing features.",
)
entropy_cols[1].metric(
    "Max entropy",
    f"{entropy_payload.get('max_entropy_bits', 0):.3f} bits" if entropy_payload else "missing",
    help="The upper bound for five equally likely classes.",
)
entropy_cols[2].metric(
    "Entropy efficiency",
    f"{entropy_payload.get('entropy_efficiency', 0):.3f}" if entropy_payload else "missing",
    help="Near 1.0 means class distribution is fairly balanced.",
)
entropy_cols[3].metric(
    "RF F1 macro",
    f"{rf_results.get('f1_macro', 0):.3f}" if rf_results else "missing",
    help="Cross-validated macro F1 for the Random Forest baseline.",
)

col_entropy, col_mi = st.columns(2)
with col_entropy:
    st.subheader("Class distribution and entropy")
    entropy_rows = entropy_class_rows(entropy_payload)
    if entropy_rows:
        st.pyplot(horizontal_bar_figure(
            entropy_rows,
            label_key="class",
            value_key="count",
            title="Samples per class",
        ))
        st.dataframe(pd.DataFrame(entropy_rows), width="stretch")
    else:
        st.info("Entropy analysis artifact not found.")

with col_mi:
    st.subheader("Mutual information ranking")
    if mi_rows:
        st.pyplot(horizontal_bar_figure(
            mi_rows,
            label_key="feature",
            value_key="mutual_information",
            title="Top mutual information features",
        ))
        st.caption("Higher values mean the feature reduces more uncertainty about stroke class.")
    else:
        st.info("Mutual information CSV not found.")

col_rf, col_tree = st.columns(2)
with col_rf:
    st.subheader("Random Forest global evidence")
    feature_importance_path = existing_path(RESULTS_ROOT / "rf_baseline" / "rf_feature_importance.png")
    if feature_importance_path:
        st.image(str(feature_importance_path), caption="RF feature importance")
    else:
        st.info("RF feature-importance figure is missing.")

with col_tree:
    st.subheader("Decision Tree interpretable baseline")
    if tree_results:
        metric_row([
            ("CV accuracy", f"{tree_results.get('cv_accuracy_mean', 0):.3f}", "Decision Tree cross-validation mean"),
            ("Tree depth", str(tree_results.get("max_depth", "?")), "Configured max depth"),
            ("Leaves", str(tree_results.get("n_leaves", "?")), "Number of terminal leaves"),
        ])
        split_rows = decision_tree_split_rows(tree_results, top_n=8)
        if split_rows:
            st.dataframe(pd.DataFrame(split_rows), width="stretch")
    else:
        st.info("Decision Tree result JSON not found.")

with st.expander("More global Data Mining figures", expanded=st.session_state.get("detail_level") == "Technical"):
    fig_cols = st.columns(3)
    figures = [
        ("RF confusion matrix", RESULTS_ROOT / "rf_baseline" / "rf_confusion_matrix.png"),
        ("Normalized RF confusion matrix", RESULTS_ROOT / "rf_baseline" / "rf_confusion_matrix_norm.png"),
        ("Mutual information", DM_RESULTS_ROOT / "mutual_information.png"),
        ("Feature correlation", DM_RESULTS_ROOT / "feature_correlation.png"),
        ("Feature distributions", DM_RESULTS_ROOT / "feature_distributions.png"),
        ("Decision tree", DM_RESULTS_ROOT / "decision_tree.png"),
    ]
    for idx, (caption, path) in enumerate(figures):
        with fig_cols[idx % len(fig_cols)]:
            found = existing_path(path)
            if found:
                st.image(str(found), caption=caption)
            else:
                st.caption(f"Missing: {caption}")

st.divider()
st.header("Local current-sample explanation")

local_cols = st.columns(2)
with local_cols[0]:
    st.subheader("Top RF features for this clip")
    top_rows = rf_top_feature_rows(rf_results, run.dm_features, top_n=12)
    if top_rows:
        st.dataframe(pd.DataFrame(top_rows), width="stretch")
    else:
        st.info("No RF top-feature metadata available.")

with local_cols[1]:
    st.subheader("Current sample vs predicted-class average")
    if rf_bundle is not None and run.rf_prediction.label:
        comparison_rows = class_average_comparison_rows(
            rf_bundle,
            run.dm_features,
            run.rf_prediction.label,
            top_n=12,
        )
        if comparison_rows:
            st.dataframe(pd.DataFrame(comparison_rows), width="stretch")
        else:
            st.info("No class-average comparison rows available.")
    elif rf_bundle_error:
        st.warning(rf_bundle_error)
    else:
        st.info("RF bundle or RF prediction is missing.")

info_gain_rows = entropy_information_gain_rows(entropy_payload, top_n=10)
if info_gain_rows and st.session_state.get("detail_level") == "Technical":
    st.subheader("Entropy information-gain details")
    st.dataframe(pd.DataFrame(info_gain_rows), width="stretch")

with st.expander("Technical: all extracted biomechanical features", expanded=False):
    feature_order = rf_bundle.feature_names if rf_bundle is not None else None
    feature_rows = feature_value_rows(run.dm_features, feature_order=feature_order)
    if feature_rows:
        st.dataframe(pd.DataFrame(feature_rows), width="stretch")
    else:
        st.info("No biomechanical feature dictionary found in this PipelineRun.")

with st.expander("Technical: decision tree rules", expanded=False):
    if tree_rules:
        st.code(tree_rules[:12000], language="text")
        if len(tree_rules) > 12000:
            st.caption("Rules truncated for display.")
    else:
        st.info("Decision Tree rules text file not found.")

st.caption(
    "Data Mining branch summary: RF is the strongest numerical classifier in this project. "
    "Decision Tree, entropy, and mutual information are shown for interpretability and theory connection."
)
