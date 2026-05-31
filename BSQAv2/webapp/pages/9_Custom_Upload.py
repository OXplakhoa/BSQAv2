from components.bootstrap import PROJECT_ROOT, ensure_project_imports
ensure_project_imports()

from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

from components.charts import probability_rows, skeleton_frame_rows
from components.dm_viz import horizontal_bar_figure
from components.skeleton_view import skeleton_figure
from components.ui import explanation_card, metric_row
from src.config import STROKE_TYPES
from src.observatory.artifacts import ArtifactRegistry
from src.observatory.quality_references import load_quality_reference_bank, reference_bank_summary
from src.observatory.schema import ArtifactValidationError
from src.observatory.upload_pipeline import run_uploaded_video_pipeline, save_uploaded_file


st.set_page_config(page_title="Custom Upload", page_icon="📤", layout="wide")

st.sidebar.title("BSQAv2 Observatory")
st.sidebar.info("Custom Upload runs live MediaPipe extraction. Use Curated pages for the stable defense path.")
st.sidebar.selectbox("Detail level", ["Simple", "Technical"], key="detail_level")
st.sidebar.toggle("Guided Presentation Mode", value=True, key="guided_mode")

st.title("Custom Video Upload")

explanation_card(
    "Live pipeline path",
    "Upload mode saves a temporary video, runs MediaPipe pose extraction, converts to COCO-17, "
    "runs pose QC and preprocessing, then runs the RF branch and Phase 4 quality scoring. "
    "DTW uses same-stroke curated references when available; DL inference is optional because CPU execution can be slow.",
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
        "Upload a badminton stroke clip",
        type=["mp4", "mov", "avi", "mkv"],
        help="Short single-stroke clips work best. Long videos can be slow because MediaPipe runs frame by frame.",
    )
    ground_truth_choice = st.selectbox("Optional ground-truth label", label_options)
    run_dl = st.checkbox(
        "Also run DL checkpoint (slower on CPU)",
        value=False,
        help="RF is fast and enabled by default. DL adds GCN+BiLSTM+Attention inference if the checkpoint is available.",
    )
    submitted = st.form_submit_button("Run upload pipeline", type="primary")

ground_truth = None if ground_truth_choice == "unknown" else ground_truth_choice

if not submitted:
    st.info("Upload a short video clip, optionally choose a ground-truth label, then run the live pipeline.")
    st.stop()

if uploaded_file is None:
    st.warning("Please upload a video file before running the pipeline.")
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
            st.warning(f"DL checkpoint not found, continuing RF-only: {candidate}")

    with st.spinner("Running MediaPipe pose extraction and model inference..."):
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
    st.error(f"Upload pipeline failed: {exc}")
    st.stop()

st.success("Upload pipeline completed.")

st.header("Prediction summary")
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
    st.subheader("RF probabilities")
    rf_rows = probability_rows(run.rf_prediction.probabilities)
    if rf_rows:
        st.pyplot(horizontal_bar_figure(rf_rows, "class", "probability", "Uploaded clip RF probabilities"))
    else:
        st.info("No RF probabilities available.")
    st.write(run.diagnostics.get("rf_summary", "No RF summary available."))
    if quality_report:
        st.write(run.diagnostics.get("quality_summary", ""))
        with st.expander("Technique quality feedback", expanded=True):
            reference_match = quality_report.get("reference_match", {})
            st.write(
                f"Reference bank: {quality_reference_summary}. "
                f"Best DTW match: {reference_match.get('best_reference_id') or 'none'} "
                f"from {reference_match.get('n_references', 0)} same-stroke reference(s)."
            )
            for item in quality_report.get("feedback", []):
                st.write(f"- {item}")

with col_pose:
    st.subheader("Pose quality")
    pose_rows = [
        {"metric": "reliability_score", "value": run.pose_qc.get("reliability_score")},
        {"metric": "reliability_label", "value": run.pose_qc.get("reliability_label")},
        {"metric": "missing_joint_ratio", "value": run.pose_qc.get("missing_joint_ratio")},
        {"metric": "max_jump", "value": run.pose_qc.get("max_jump")},
        {"metric": "outlier_jump_count", "value": run.pose_qc.get("outlier_jump_count")},
    ]
    st.dataframe(pd.DataFrame(pose_rows).astype(str), width="stretch")
    if run.pose_qc.get("warnings"):
        st.warning("Pose warnings: " + "; ".join(run.pose_qc["warnings"]))
    else:
        st.success("No pose warnings for this upload.")

st.header("Extracted skeleton preview")
raw = run.arrays.get("raw_keypoints")
if raw is not None and raw.size:
    frame_idx = st.slider("Frame", min_value=0, max_value=int(raw.shape[0] - 1), value=int(raw.shape[0] // 2))
    fig = skeleton_figure(raw, frame_index=frame_idx)
    st.pyplot(fig)
    if st.session_state.get("detail_level") == "Technical":
        st.dataframe(pd.DataFrame(skeleton_frame_rows(run, frame_idx)), width="stretch")
else:
    st.info("No raw keypoints available for skeleton preview.")

if run.dl_prediction.label:
    st.header("Optional DL result")
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
