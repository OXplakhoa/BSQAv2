from components.bootstrap import ensure_project_imports
ensure_project_imports()

import pandas as pd
import streamlit as st

from components.charts import attention_frame_importance, probability_rows
from components.data import load_selected_case
from components.dl_viz import (
    attention_heatmap_figure,
    confidence_interpretation,
    dl_shape_rows,
    top_attention_frames,
)
from components.ui import explanation_card, metric_row, render_glossary_footer, render_sidebar, show_video


st.set_page_config(page_title="Deep Learning Inspector — Giải thích DL", page_icon="🧠", layout="wide")

sample_id = render_sidebar()
st.title("Deep Learning Inspector — Giải thích DL")

explanation_card(
    "GCN + BiLSTM + Temporal Attention",
    "Nhánh DL nhận chuỗi skeleton normalized `(64, 17, 2)`. GCN học quan hệ giữa các khớp "
    "trong từng frame, BiLSTM học chuyển động theo thời gian, và temporal Attention làm nổi bật "
    "những frame ảnh hưởng mạnh tới dự đoán cuối.",
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
        ("Ground truth", sample.ground_truth or "unknown", "Manual curated label"),
        ("DL prediction", run.dl_prediction.label or "missing", "GCN + BiLSTM + Attention"),
        ("DL confidence", f"{(run.dl_prediction.confidence or 0):.3f}", "Softmax top-class probability"),
        ("RF prediction", run.rf_prediction.label or "missing", "Comparison branch"),
    ])
    st.write(confidence_interpretation(run.dl_prediction))

st.subheader("Đường đi kiến trúc mô hình")
st.markdown(
    """
```text
Normalized skeleton (64 frames, 17 joints, x/y)
  -> velocity channels added internally
  -> Spatial GCN per frame
  -> BiLSTM temporal sequence model
  -> Temporal self-attention
  -> Classifier head -> stroke probabilities
```
"""
)

if st.session_state.get("detail_level") == "Technical":
    st.subheader("Tensor shapes")
    shape_rows = dl_shape_rows(run)
    if shape_rows:
        st.dataframe(pd.DataFrame(shape_rows), width="stretch")
    else:
        st.info("Không có metadata tensor shape trong PipelineRun này.")

col_prob, col_attention = st.columns(2)
with col_prob:
    st.subheader("Xác suất class của DL")
    prob_rows = probability_rows(run.dl_prediction.probabilities)
    if prob_rows:
        st.bar_chart(pd.DataFrame(prob_rows).set_index("class"))
    else:
        st.info("Không có dữ liệu xác suất DL.")

with col_attention:
    st.subheader("Độ quan trọng từng frame")
    attention_rows = attention_frame_importance(run)
    if attention_rows:
        st.line_chart(pd.DataFrame(attention_rows).set_index("frame"))
    else:
        st.info("Không có attention weights.")

st.subheader("Các frame Attention cao nhất")
top_rows = top_attention_frames(run)
if top_rows:
    st.dataframe(pd.DataFrame(top_rows), width="stretch")
else:
    st.info("Không có danh sách frame attention cao nhất.")

attention = run.arrays.get("attention_weights")
if attention is not None and attention.size:
    with st.expander("Technical: attention heatmap", expanded=st.session_state.get("detail_level") == "Technical"):
        fig = attention_heatmap_figure(attention)
        st.pyplot(fig)

st.subheader("Ngữ cảnh RF vs DL")
st.write(run.diagnostics.get("branch_comparison", "No branch comparison available."))
st.caption(
    "For this project, RF is the stronger numerical classifier on the curated set. "
    "The DL branch is still important because it demonstrates spatial graph learning, "
    "temporal sequence modeling, and attention-based inspection."
)

render_glossary_footer()
