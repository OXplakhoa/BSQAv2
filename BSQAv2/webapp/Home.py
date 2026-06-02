from components.bootstrap import ensure_project_imports
ensure_project_imports()

import pandas as pd
import streamlit as st

from components.charts import prediction_summary_rows
from components.data import load_selected_case
from components.ui import explanation_card, metric_row, render_glossary_footer, render_sidebar, render_three_hero, show_video


st.set_page_config(page_title="BSQAv2 Pipeline Observatory", page_icon="🏸", layout="wide")

sample_id = render_sidebar()

render_three_hero()

st.markdown("## BSQAv2 Pipeline Observatory — Phòng quan sát pipeline")
st.caption("Video -> MediaPipe Pose -> COCO-17 Skeleton -> RF + GCN/BiLSTM/Attention")

explanation_card(
    "Ứng dụng này chứng minh điều gì?",
    "Đây không chỉ là một classifier. App giải thích cách video cầu lông trở thành chuỗi pose, "
    "pose quality ảnh hưởng dự đoán ra sao, và vì sao RF và DL có thể đồng ý hoặc khác nhau.",
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
    st.subheader("Mẫu đang chọn")
    metric_row([
        ("Ground truth", sample.ground_truth or "unknown", "Manual curated label"),
        ("Pose reliability", f"{run.pose_qc.get('reliability_score', 0):.2f}", run.pose_qc.get("reliability_label", "unknown")),
        ("RF prediction", run.rf_prediction.label or "missing", "Random Forest branch"),
        ("DL prediction", run.dl_prediction.label or "missing", "GCN + BiLSTM + Attention branch"),
    ])

    st.write("**Điểm cần quan sát (Teaching point)**")
    st.write(sample.teaching_point)
    st.write("**Chẩn đoán (Diagnosis)**")
    st.write(sample.diagnosis)

st.subheader("Tóm tắt hai nhánh mô hình")
st.dataframe(pd.DataFrame(prediction_summary_rows(run)), width="stretch")

if run.pose_qc.get("warnings"):
    st.warning("Cảnh báo pose: " + "; ".join(run.pose_qc["warnings"]))
else:
    st.success("Không có cảnh báo pose cho mẫu này.")

st.subheader("Nên mở trang nào tiếp theo?")
st.markdown(
    """
- **Full Pipeline Demo**: xem pipeline từng bước cho mẫu đang chọn.
- **Pose Inspector**: kiểm tra visibility, missing joints, và skeleton theo từng frame.
- **Deep Learning Inspector**: xác suất GCN + BiLSTM + Attention, tensor shapes, và attention maps.
- **Data Mining Motion Lab**: biomechanical features, bằng chứng RF, entropy, mutual information, và Decision Tree rules.
- **Error Analysis Lab**: so sánh RF-vs-DL, confidence margin, pose risk, và các ca dễ nhầm.
- **Training & Evaluation**: metrics RF/DL, per-class F1, confusion matrix, và checkpoint inventory.
- **Dataset Explorer**: phân bố source/class, curated manifest, và pose-reliability summary.
- **Robustness Experiment**: thêm nhiễu skeleton nhân tạo và xem RF sensitivity.
- **Custom Upload**: trang beta chạy MediaPipe + RF trên clip ngắn của người dùng.

Custom Upload là **beta**. Khi demo live, dùng curated investigation để ổn định nhất; upload chậm hơn vì MediaPipe chạy từng frame.
"""
)

render_glossary_footer()
