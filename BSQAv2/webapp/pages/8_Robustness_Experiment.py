from components.bootstrap import ensure_project_imports
ensure_project_imports()

import pandas as pd
import streamlit as st

from components.charts import probability_rows
from components.data import load_selected_case
from components.dm_viz import horizontal_bar_figure
from components.robustness_viz import (
    degradation_summary_rows,
    degrade_keypoints,
    prediction_delta_rows,
    robustness_curve_figure,
    robustness_curve_rows,
)
from components.skeleton_view import skeleton_figure
from components.ui import explanation_card, metric_row, render_sidebar, show_video
from src.data.biomechanics import extract_features
from src.observatory.artifacts import ArtifactRegistry
from src.observatory.dm_inference import load_rf_bundle
from src.observatory.schema import ArtifactValidationError, PredictionResult


st.set_page_config(page_title="Robustness Experiment", page_icon="🧪", layout="wide")

sample_id = render_sidebar()
st.title("Robustness Experiment")

explanation_card(
    "How sensitive is the pipeline to pose degradation?",
    "This lab perturbs the selected cached skeleton with artificial noise, joint dropout, and frame dropout. "
    "It reruns the fast Data Mining/RF branch on degraded features to show how upstream pose quality can change downstream confidence. "
    "The cached DL prediction is shown as a baseline reference; live DL re-inference is intentionally avoided here for presentation speed.",
)

if sample_id is None:
    st.stop()

case = load_selected_case(sample_id)
sample = case.sample
run = case.run
normalized = run.arrays.get("normalized_keypoints")
if normalized is None:
    st.error("Selected PipelineRun has no normalized_keypoints array.")
    st.stop()

left, right = st.columns([1.0, 1.1])
with left:
    st.subheader(sample.title)
    show_video(sample.video_path)
with right:
    st.subheader("Baseline predictions")
    metric_row([
        ("Ground truth", sample.ground_truth or "unknown", "Manual curated label"),
        ("Baseline RF", run.rf_prediction.label or "missing", "Cached RF prediction"),
        ("RF confidence", f"{(run.rf_prediction.confidence or 0):.3f}", "Cached RF top-class probability"),
        ("Cached DL", run.dl_prediction.label or "missing", "Cached DL reference prediction"),
    ])

st.divider()
st.header("Degradation controls")

ctrl_cols = st.columns(4)
with ctrl_cols[0]:
    noise_std = st.slider("Coordinate noise σ", min_value=0.0, max_value=0.30, value=0.05, step=0.01)
with ctrl_cols[1]:
    frame_dropout_rate = st.slider("Frame dropout", min_value=0.0, max_value=0.50, value=0.10, step=0.05)
with ctrl_cols[2]:
    drop_wrists = st.checkbox("Drop wrists", value=False)
    drop_elbows = st.checkbox("Drop elbows", value=False)
with ctrl_cols[3]:
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

registry = ArtifactRegistry()
try:
    rf_bundle = load_rf_bundle(registry.resolve_rf_bundle())
except ArtifactValidationError as exc:
    st.error(str(exc))
    st.stop()


def _predict_rf(keypoints) -> PredictionResult:
    features = extract_features(keypoints)
    return rf_bundle.predict(features)


degraded = degrade_keypoints(
    normalized,
    noise_std=noise_std,
    drop_wrists=drop_wrists,
    drop_elbows=drop_elbows,
    frame_dropout_rate=frame_dropout_rate,
    seed=int(seed),
)
degraded_prediction = _predict_rf(degraded)
summary_rows = degradation_summary_rows(normalized, degraded)
delta_rows = prediction_delta_rows(run.rf_prediction.probabilities, degraded_prediction.probabilities)

st.subheader("Degraded RF result")
metric_row([
    ("Degraded RF", degraded_prediction.label or "missing", "RF prediction after artificial degradation"),
    ("Degraded confidence", f"{(degraded_prediction.confidence or 0):.3f}", "Top-class probability after degradation"),
    ("Label changed?", "yes" if degraded_prediction.label != run.rf_prediction.label else "no", "Whether degradation changed RF top class"),
    ("Confidence delta", f"{((degraded_prediction.confidence or 0) - (run.rf_prediction.confidence or 0)):.3f}", "Degraded confidence minus baseline confidence"),
])

chart_cols = st.columns(2)
with chart_cols[0]:
    st.subheader("Baseline RF probabilities")
    baseline_rows = probability_rows(run.rf_prediction.probabilities)
    if baseline_rows:
        st.pyplot(horizontal_bar_figure(baseline_rows, "class", "probability", "Baseline RF probabilities"))
with chart_cols[1]:
    st.subheader("Degraded RF probabilities")
    degraded_rows = probability_rows(degraded_prediction.probabilities)
    if degraded_rows:
        st.pyplot(horizontal_bar_figure(degraded_rows, "class", "probability", "Degraded RF probabilities"))

st.subheader("Probability changes")
if delta_rows:
    st.dataframe(pd.DataFrame(delta_rows), width="stretch")
else:
    st.info("No probability rows available for delta comparison.")

st.divider()
st.header("Pose degradation evidence")

pose_cols = st.columns(2)
with pose_cols[0]:
    st.subheader("Degradation summary")
    st.dataframe(pd.DataFrame(summary_rows), width="stretch")
with pose_cols[1]:
    st.subheader("Skeleton preview")
    frame_index = st.slider("Preview frame", min_value=0, max_value=int(normalized.shape[0] - 1), value=int(normalized.shape[0] // 2))

fig_cols = st.columns(2)
with fig_cols[0]:
    st.pyplot(skeleton_figure(normalized, frame_index=frame_index))
with fig_cols[1]:
    st.pyplot(skeleton_figure(degraded, frame_index=frame_index))

st.divider()
st.header("Noise sensitivity curve")

curve_results = []
for severity in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
    perturbed = degrade_keypoints(
        normalized,
        noise_std=severity,
        drop_wrists=drop_wrists,
        drop_elbows=drop_elbows,
        frame_dropout_rate=frame_dropout_rate,
        seed=int(seed),
    )
    pred = _predict_rf(perturbed)
    curve_results.append({
        "severity": severity,
        "prediction": pred.label,
        "confidence": pred.confidence or 0.0,
    })
curve_rows = robustness_curve_rows(curve_results)
curve_df = pd.DataFrame(curve_rows)
st.dataframe(curve_df, width="stretch")
st.pyplot(robustness_curve_figure(curve_rows))
st.caption(
    "This curve varies coordinate noise while keeping the selected joint/frame dropout settings fixed. "
    "It demonstrates downstream sensitivity rather than a formal adversarial robustness benchmark."
)

if st.session_state.get("detail_level") == "Technical":
    with st.expander("Technical: degraded RF probabilities", expanded=True):
        st.json(degraded_prediction.probabilities)

st.caption(
    "Robustness scope: cached skeleton perturbation + RF re-inference. Future work can cache DL degraded inference curves for faster live presentation."
)
