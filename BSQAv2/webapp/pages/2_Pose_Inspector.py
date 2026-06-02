from components.bootstrap import ensure_project_imports
ensure_project_imports()

import pandas as pd
import streamlit as st

from components.charts import critical_visibility_rows, missing_joint_ratio_rows, skeleton_frame_rows
from components.data import load_selected_case
from components.skeleton_view import skeleton_figure
from components.ui import explanation_card, metric_row, render_glossary_footer, render_sidebar, show_video


st.set_page_config(page_title="Pose Inspector — Kiểm tra skeleton", page_icon="🏸", layout="wide")

sample_id = render_sidebar()
st.title("Pose Inspector — Kiểm tra skeleton")

explanation_card(
    "Chẩn đoán pose",
    "Skeleton theo pixel gốc dùng để kiểm tra độ tin cậy MediaPipe. Nhánh model dùng "
    "skeleton normalized `(64, 17, 2)` riêng, lưu trong cùng PipelineRun artifact.",
)

if sample_id is None:
    st.stop()

case = load_selected_case(sample_id)
sample = case.sample
run = case.run
raw = run.arrays.get("raw_keypoints")

left, right = st.columns([1.1, 1])
with left:
    show_video(sample.video_path)
with right:
    st.subheader("Chất lượng pose (Pose quality)")
    metric_row([
        ("Reliability", f"{run.pose_qc.get('reliability_score', 0):.2f}", run.pose_qc.get("reliability_label", "unknown")),
        ("Missing ratio", f"{run.pose_qc.get('missing_ratio', 0):.1%}", "Overall missing/low-visibility joint ratio"),
        ("Max jump", f"{run.pose_qc.get('max_jump_px', 0):.0f}px", "Largest valid frame-to-frame joint displacement"),
        ("Outlier jumps", run.pose_qc.get("outlier_jump_count", 0), "Jumps above threshold"),
    ])
    if run.pose_qc.get("warnings"):
        st.warning("\n".join(run.pose_qc["warnings"]))
    else:
        st.success("Không có cảnh báo pose.")

st.subheader("Tỉ lệ missing joint theo thời gian")
missing_rows = missing_joint_ratio_rows(run)
if missing_rows:
    st.line_chart(pd.DataFrame(missing_rows).set_index("frame"))
else:
    st.info("Không có chuỗi thời gian missing-joint.")

st.subheader("Visibility của khớp quan trọng")
visibility_rows = critical_visibility_rows(run)
if visibility_rows:
    st.bar_chart(pd.DataFrame(visibility_rows).set_index("joint"))
else:
    st.info("Không có dữ liệu visibility cho khớp quan trọng.")

if raw is not None:
    st.subheader("Bảng keypoint theo frame")
    frame = st.slider("Frame", 0, int(raw.shape[0] - 1), int(raw.shape[0] // 2))
    rows = skeleton_frame_rows(run, frame_index=frame, normalized=False)
    fig = skeleton_figure(raw, frame)
    st.pyplot(fig)
    st.dataframe(pd.DataFrame(rows), width="stretch")

render_glossary_footer()
