from components.bootstrap import ensure_project_imports
ensure_project_imports()

import pandas as pd
import streamlit as st

from components.charts import attention_frame_importance, probability_rows
from components.data import load_selected_case
from components.ui import explanation_card, metric_row, render_sidebar, show_video


st.set_page_config(page_title="Full Pipeline Demo", page_icon="🏸", layout="wide")

sample_id = render_sidebar()
st.title("Full Pipeline Demo")

explanation_card(
    "Pipeline walkthrough",
    "This page follows one curated clip through the cached pipeline artifact: video metadata, "
    "pose quality, normalized skeleton, RF prediction, DL prediction, and branch comparison.",
)

if sample_id is None:
    st.stop()

case = load_selected_case(sample_id)
sample = case.sample
run = case.run

left, right = st.columns([1.1, 1])
with left:
    show_video(sample.video_path)
with right:
    st.subheader(sample.title)
    metric_row([
        ("Truth", sample.ground_truth or "unknown", "Ground-truth label"),
        ("Raw frames", run.video_metadata.get("raw_frame_count", "?"), "Frames extracted from clip"),
        ("Normalized shape", str(list(run.arrays["normalized_keypoints"].shape)), "Model input tensor"),
        ("Pose score", f"{run.pose_qc.get('reliability_score', 0):.2f}", run.pose_qc.get("reliability_label", "unknown")),
    ])

st.subheader("Pipeline stages")
st.progress(1.0)
st.markdown(
    """
1. Video clip selected from curated manifest
2. MediaPipe pose extracted and cached
3. COCO-17 raw keypoints stored for pose diagnostics
4. Skeleton normalized to `(64, 17, 2)`
5. Biomechanical features extracted for RF
6. GCN + BiLSTM + Attention run on normalized skeleton
7. Diagnostics generated from predictions and pose quality
"""
)

col_rf, col_dl = st.columns(2)
with col_rf:
    st.subheader("Random Forest")
    st.bar_chart(pd.DataFrame(probability_rows(run.rf_prediction.probabilities)).set_index("class"))
    st.write(run.diagnostics.get("rf_summary", "No RF summary available."))

with col_dl:
    st.subheader("Deep Learning")
    st.bar_chart(pd.DataFrame(probability_rows(run.dl_prediction.probabilities)).set_index("class"))
    st.write(run.diagnostics.get("dl_summary", "No DL summary available."))

st.subheader("Branch comparison")
st.write(run.diagnostics.get("branch_comparison", "No branch comparison available."))

attention_rows = attention_frame_importance(run)
if attention_rows:
    st.subheader("DL frame importance")
    st.line_chart(pd.DataFrame(attention_rows).set_index("frame"))
