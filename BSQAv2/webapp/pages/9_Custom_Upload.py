from components.bootstrap import PROJECT_ROOT, ensure_project_imports
ensure_project_imports()

from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

from components.charts import probability_rows, skeleton_frame_rows
from components.dm_viz import horizontal_bar_figure
from components.skeleton_view import skeleton_figure
from components.ui import explanation_card, inject_design_system, metric_row, render_glossary_footer
from src.config import STROKE_TYPES
from src.observatory.artifacts import ArtifactRegistry
from src.observatory.quality_references import load_quality_reference_bank, reference_bank_summary
from src.observatory.schema import ArtifactValidationError
from src.observatory.upload_pipeline import run_uploaded_video_pipeline, save_uploaded_file


st.set_page_config(page_title="Custom Upload", page_icon="📤", layout="wide")
inject_design_system()

st.sidebar.title("BSQAv2 Observatory")
st.sidebar.info("Custom Upload chạy MediaPipe trực tiếp. Khi demo quan trọng, dùng Curated pages để ổn định hơn.")
st.sidebar.selectbox("Mức chi tiết", ["Simple", "Technical"], format_func={"Simple": "Dễ hiểu", "Technical": "Kỹ thuật"}.get, key="detail_level")
st.sidebar.toggle("Chế độ thuyết trình có hướng dẫn", value=True, key="guided_mode")

st.title("Custom Upload — Tải video của bạn")

explanation_card(
    "Đường đi pipeline trực tiếp",
    "Upload mode lưu video tạm, chạy MediaPipe pose extraction, chuyển sang COCO-17, "
    "chạy Pose QC và preprocessing, sau đó chạy nhánh RF và quality scoring Phase 4. "
    "DTW dùng reference curated cùng loại cú đánh nếu có; DL inference là tùy chọn vì CPU có thể chậm.",
)

registry = ArtifactRegistry()
upload_dir = PROJECT_ROOT / "webapp" / "artifacts" / "uploads"


@st.cache_resource(show_spinner=False)
def _cached_quality_reference_bank():
    return load_quality_reference_bank(max_per_stroke=3, min_pose_reliability=0.0)


quality_reference_bank = _cached_quality_reference_bank()
quality_reference_summary = reference_bank_summary(quality_reference_bank)

label_options = ["unknown"] + list(STROKE_TYPES)
with st.form("custom_upload_form"):
    uploaded_file = st.file_uploader(
        "Tải clip cú đánh cầu lông",
        type=["mp4", "mov", "avi", "mkv"],
        help="Clip ngắn một cú đánh là tốt nhất. Video dài sẽ chậm vì MediaPipe chạy từng frame.",
    )
    ground_truth_choice = st.selectbox("Nhãn đúng tùy chọn (ground truth)", label_options)
    run_dl = st.checkbox(
        "Chạy thêm DL checkpoint (chậm hơn trên CPU)",
        value=False,
        help="RF nhanh và luôn chạy. DL thêm inference GCN+BiLSTM+Attention nếu có checkpoint.",
    )
    submitted = st.form_submit_button("Chạy upload pipeline", type="primary")

ground_truth = None if ground_truth_choice == "unknown" else ground_truth_choice

if not submitted:
    st.info("Tải một clip ngắn, có thể chọn ground-truth label, rồi chạy live pipeline.")
    st.stop()

if uploaded_file is None:
    st.warning("Vui lòng tải video trước khi chạy pipeline.")
    st.stop()

st.video(uploaded_file.getvalue())

try:
    saved_path = save_uploaded_file(uploaded_file.getvalue(), upload_dir, uploaded_file.name)
    rf_bundle_path = registry.resolve_rf_bundle()
    dl_checkpoint_path = None
    if run_dl:
        candidate = PROJECT_ROOT / "_colab_results" / "gcn_bilstm_attn_20260528_095136" / "best_model_fold3.pth"
        if candidate.exists():
            dl_checkpoint_path = candidate
        else:
            st.warning(f"Không tìm thấy DL checkpoint, tiếp tục RF-only: {candidate}")

    with st.spinner("Đang chạy MediaPipe pose extraction và model inference..."):
        run = run_uploaded_video_pipeline(
            saved_path,
            sample_id=f"upload_{uuid4().hex[:8]}",
            ground_truth=ground_truth,
            rf_bundle_path=rf_bundle_path,
            dl_checkpoint_path=dl_checkpoint_path,
            dl_device="cpu",
            quality_references=quality_reference_bank,
        )
except (ArtifactValidationError, FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
    st.error(f"Upload pipeline thất bại: {exc}")
    st.stop()

st.success("Upload pipeline đã hoàn tất.")

st.header("Tóm tắt dự đoán")
quality_report = run.diagnostics.get("quality_report", {})
metrics = [
    ("Ground truth", ground_truth or "unknown", "Optional user-provided label"),
    ("RF prediction", run.rf_prediction.label or "missing", "Random Forest on extracted biomechanical features"),
    ("RF confidence", f"{(run.rf_prediction.confidence or 0):.3f}", "Top RF probability"),
    ("Pose reliability", f"{run.pose_qc.get('reliability_score', 0):.3f}", run.pose_qc.get("reliability_label", "unknown")),
]
if quality_report:
    metrics.append(("Technique quality", f"{quality_report.get('quality_score', 0):.0f}/100", "Phase 4 hybrid quality estimate"))
    if quality_report.get("dtw_score") is not None:
        metrics.append(("DTW similarity", f"{quality_report.get('dtw_score', 0):.0f}/100", "Compared with same-stroke curated references"))
if run.dl_prediction.label:
    metrics.append(("DL prediction", run.dl_prediction.label, "Optional GCN + BiLSTM + Attention branch"))
metric_row(metrics)

col_rf, col_pose = st.columns(2)
with col_rf:
    st.subheader("Xác suất RF")
    rf_rows = probability_rows(run.rf_prediction.probabilities)
    if rf_rows:
        st.pyplot(horizontal_bar_figure(rf_rows, "class", "probability", "Uploaded clip RF probabilities"))
    else:
        st.info("Không có xác suất RF.")
    st.write(run.diagnostics.get("rf_summary", "No RF summary available."))
    if quality_report:
        st.write(run.diagnostics.get("quality_summary", ""))
        with st.expander("Phản hồi chất lượng kỹ thuật", expanded=True):
            reference_match = quality_report.get("reference_match", {})
            st.write(
                f"Reference bank: {quality_reference_summary}. "
                f"Best DTW match: {reference_match.get('best_reference_id') or 'none'} "
                f"from {reference_match.get('n_references', 0)} same-stroke reference(s)."
            )
            for item in quality_report.get("feedback", []):
                st.write(f"- {item}")

with col_pose:
    st.subheader("Chất lượng pose (Pose quality)")
    pose_rows = [
        {"metric": "reliability_score", "value": run.pose_qc.get("reliability_score")},
        {"metric": "reliability_label", "value": run.pose_qc.get("reliability_label")},
        {"metric": "missing_joint_ratio", "value": run.pose_qc.get("missing_joint_ratio")},
        {"metric": "max_jump", "value": run.pose_qc.get("max_jump")},
        {"metric": "outlier_jump_count", "value": run.pose_qc.get("outlier_jump_count")},
    ]
    st.dataframe(pd.DataFrame(pose_rows).astype(str), width="stretch")
    if run.pose_qc.get("warnings"):
        st.warning("Cảnh báo pose: " + "; ".join(run.pose_qc["warnings"]))
    else:
        st.success("Không có cảnh báo pose cho upload này.")

st.header("Xem trước skeleton đã trích xuất")
raw = run.arrays.get("raw_keypoints")
if raw is not None and raw.size:
    frame_idx = st.slider("Frame", min_value=0, max_value=int(raw.shape[0] - 1), value=int(raw.shape[0] // 2))
    fig = skeleton_figure(raw, frame_index=frame_idx)
    st.pyplot(fig)
    if st.session_state.get("detail_level") == "Technical":
        st.dataframe(pd.DataFrame(skeleton_frame_rows(run, frame_idx)), width="stretch")
else:
    st.info("Không có raw keypoints để xem skeleton.")

if run.dl_prediction.label:
    st.header("Kết quả DL tùy chọn")
    dl_rows = probability_rows(run.dl_prediction.probabilities)
    if dl_rows:
        st.pyplot(horizontal_bar_figure(dl_rows, "class", "probability", "Uploaded clip DL probabilities"))
    st.write(run.diagnostics.get("dl_summary", "No DL summary available."))

with st.expander("Technical: upload PipelineRun metadata", expanded=st.session_state.get("detail_level") == "Technical"):
    st.json({
        "run_id": run.run_id,
        "sample_id": run.sample_id,
        "mode": run.mode,
        "source_video_path": run.source_video_path,
        "video_metadata": run.video_metadata,
        "diagnostics": run.diagnostics,
        "array_shapes": {name: list(value.shape) for name, value in run.arrays.items()},
    })

st.caption(
    "Upload mode is a beta live-processing path. It is useful for authenticity, but curated mode is safer for live presentation."
)

render_glossary_footer()
