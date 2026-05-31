from components.bootstrap import ensure_project_imports
ensure_project_imports()

import pandas as pd
import streamlit as st

from components.charts import prediction_summary_rows
from components.data import load_selected_case
from components.ui import explanation_card, metric_row, render_sidebar, show_video


st.set_page_config(page_title="BSQAv2 Pipeline Observatory", page_icon="🏸", layout="wide")

sample_id = render_sidebar()

st.title("BSQAv2 Pipeline Observatory")
st.caption("Video -> MediaPipe Pose -> COCO-17 Skeleton -> RF + GCN/BiLSTM/Attention")

explanation_card(
    "What this app demonstrates",
    "This is not just a classifier. It explains how badminton videos become pose "
    "sequences, how pose quality affects predictions, and why RF and DL may agree or disagree.",
)

if sample_id is None:
    st.stop()

case = load_selected_case(sample_id)
sample = case.sample
run = case.run

left, right = st.columns([1.2, 1])
with left:
    st.subheader(sample.title)
    show_video(sample.video_path)

with right:
    st.subheader("Selected case")
    metric_row([
        ("Ground truth", sample.ground_truth or "unknown", "Manual curated label"),
        ("Pose reliability", f"{run.pose_qc.get('reliability_score', 0):.2f}", run.pose_qc.get("reliability_label", "unknown")),
        ("RF prediction", run.rf_prediction.label or "missing", "Random Forest branch"),
        ("DL prediction", run.dl_prediction.label or "missing", "GCN + BiLSTM + Attention branch"),
    ])

    st.write("**Teaching point**")
    st.write(sample.teaching_point)
    st.write("**Diagnosis**")
    st.write(sample.diagnosis)

st.subheader("Model branch summary")
st.dataframe(pd.DataFrame(prediction_summary_rows(run)), width="stretch")

if run.pose_qc.get("warnings"):
    st.warning("Pose warnings: " + "; ".join(run.pose_qc["warnings"]))
else:
    st.success("No pose warnings for this case.")

st.subheader("What to open next")
st.markdown(
    """
- **Full Pipeline Demo**: step-by-step selected case summary.
- **Pose Inspector**: visibility, missing joints, and skeleton frame inspection.
- **Deep Learning Inspector**: GCN + BiLSTM + Attention probabilities, tensor shapes, and attention maps.
- **Data Mining Motion Lab**: biomechanical features, RF evidence, entropy, mutual information, and Decision Tree rules.
- **Error Analysis Lab**: RF-vs-DL agreement, confidence margins, pose risks, and curated ambiguity cases.
- **Training & Evaluation**: final RF/DL metrics, per-class F1, confusion matrices, and checkpoint inventory.
- **Dataset Explorer**: source/class distributions, curated manifest, and pose-reliability summary.
- **Robustness Experiment**: artificial pose degradation and RF sensitivity analysis.
- **Custom Upload**: beta live upload path that runs MediaPipe + RF on a short user clip.

Custom upload is available as a **beta** page. Use curated investigation pages for the safest live-defense flow; upload mode is slower because it runs MediaPipe frame-by-frame.
"""
)
