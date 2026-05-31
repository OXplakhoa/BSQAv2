from components.bootstrap import ensure_project_imports
ensure_project_imports()

import pandas as pd
import streamlit as st

from components.charts import probability_rows
from components.data import load_curated_samples, load_selected_case
from components.dm_viz import horizontal_bar_figure
from components.error_viz import (
    branch_agreement_label,
    curated_error_case_rows,
    known_confusion_note,
    pose_risk_rows,
    prediction_audit_rows,
    probability_comparison_rows,
)
from components.ui import explanation_card, metric_row, render_sidebar, show_video
from src.observatory.diagnostics import confidence_margin


st.set_page_config(page_title="Error Analysis Lab", page_icon="🔍", layout="wide")

sample_id = render_sidebar()
st.title("Error Analysis Lab")

explanation_card(
    "Turn errors into evidence",
    "This page does not hide wrong or uncertain predictions. It compares RF and DL branches, "
    "checks confidence margins, surfaces pose-quality risks, and explains likely class-confusion causes. "
    "For uploaded clips without ground truth, the page must stay cautious and avoid correctness claims.",
)

if sample_id is None:
    st.stop()

case = load_selected_case(sample_id)
sample = case.sample
run = case.run
ground_truth = sample.ground_truth or run.ground_truth

rf_status = "unknown"
dl_status = "unknown"
if ground_truth:
    rf_status = "correct" if run.rf_prediction.label == ground_truth else "incorrect"
    dl_status = "correct" if run.dl_prediction.label == ground_truth else "incorrect"

agreement = branch_agreement_label(run)
pose_label = run.pose_qc.get("reliability_label", "unknown")
pose_score = run.pose_qc.get("reliability_score")

left, right = st.columns([1.05, 1])
with left:
    st.subheader(sample.title)
    show_video(sample.video_path)
with right:
    st.subheader("Selected-case audit")
    metric_row([
        ("Ground truth", ground_truth or "unknown", "No correctness claim is made if missing"),
        ("RF status", rf_status, "RF prediction compared with ground truth"),
        ("DL status", dl_status, "DL prediction compared with ground truth"),
        ("Branch agreement", agreement, "Whether RF and DL predict the same label"),
    ])
    metric_row([
        ("RF confidence", f"{(run.rf_prediction.confidence or 0):.3f}", "RF top class probability"),
        ("DL confidence", f"{(run.dl_prediction.confidence or 0):.3f}", "DL top class probability"),
        ("Pose reliability", f"{pose_score:.3f}" if pose_score is not None else "unknown", pose_label),
        ("Pose warnings", str(len(run.pose_qc.get("warnings", []))), "Upstream pose-quality warnings"),
    ])

if not ground_truth:
    st.warning(
        "Ground truth is unknown for this case. The app reports predictions, confidence, and pose quality, "
        "but does not mark the prediction as correct or wrong."
    )

st.divider()
st.header("Prediction comparison")

col_audit, col_probs = st.columns([1, 1.1])
with col_audit:
    st.subheader("Correctness and confidence margin")
    audit_rows = prediction_audit_rows(run, ground_truth)
    st.dataframe(pd.DataFrame(audit_rows), width="stretch")

    rf_margin = confidence_margin(run.rf_prediction.probabilities)
    dl_margin = confidence_margin(run.dl_prediction.probabilities)
    if rf_margin is not None or dl_margin is not None:
        st.write(
            f"RF margin: **{rf_margin:.3f}**" if rf_margin is not None else "RF margin unavailable",
        )
        st.write(
            f"DL margin: **{dl_margin:.3f}**" if dl_margin is not None else "DL margin unavailable",
        )
        st.caption("Small top-1/top-2 margin suggests an ambiguous case even when the final label is correct.")

with col_probs:
    st.subheader("RF vs DL probability gap")
    probability_rows_merged = probability_comparison_rows(run)
    if probability_rows_merged:
        st.dataframe(pd.DataFrame(probability_rows_merged), width="stretch")
    else:
        st.info("No probability dictionaries available for branch comparison.")

chart_cols = st.columns(2)
with chart_cols[0]:
    rf_rows = probability_rows(run.rf_prediction.probabilities)
    if rf_rows:
        st.pyplot(horizontal_bar_figure(rf_rows, "class", "probability", "RF probabilities"))
with chart_cols[1]:
    dl_rows = probability_rows(run.dl_prediction.probabilities)
    if dl_rows:
        st.pyplot(horizontal_bar_figure(dl_rows, "class", "probability", "DL probabilities"))

st.divider()
st.header("Likely error factors")

factor_cols = st.columns(2)
with factor_cols[0]:
    st.subheader("Pose quality contribution")
    risk_rows = pose_risk_rows(run)
    st.dataframe(pd.DataFrame(risk_rows), width="stretch")
    if run.pose_qc.get("warnings"):
        st.warning("Pose warnings: " + "; ".join(run.pose_qc.get("warnings", [])))
    else:
        st.success("No pose warnings for this selected case.")

with factor_cols[1]:
    st.subheader("Class-confusion note")
    predicted_for_note = run.dl_prediction.label if dl_status == "incorrect" else run.rf_prediction.label
    st.write(known_confusion_note(ground_truth, predicted_for_note))
    st.subheader("Automatic branch diagnosis")
    st.write(run.diagnostics.get("branch_comparison", "No branch-comparison diagnostic available."))
    st.write(run.diagnostics.get("rf_summary", "No RF summary available."))
    st.write(run.diagnostics.get("dl_summary", "No DL summary available."))

with st.expander("Curated error/ambiguity cases", expanded=True):
    rows = curated_error_case_rows(load_curated_samples(), selected_sample_id=sample_id)
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch")
        st.caption("Use the sidebar sample selector to jump to one of these teaching cases.")
    else:
        st.info("No curated error/ambiguity cases found in the manifest tags.")

if st.session_state.get("detail_level") == "Technical":
    with st.expander("Technical: raw diagnostics payload", expanded=False):
        st.json(run.diagnostics)

st.caption(
    "Error analysis is intentionally cautious: wrong predictions can come from pose tracking, feature overlap, "
    "small confidence margins, missing racket/shuttle context, or model limitations."
)
