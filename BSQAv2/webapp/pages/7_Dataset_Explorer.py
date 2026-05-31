from components.bootstrap import PROJECT_ROOT, ensure_project_imports
ensure_project_imports()

import pandas as pd
import streamlit as st

from components.dataset_viz import (
    class_distribution_rows,
    curated_sample_rows,
    dataset_csv_summaries,
    label_quality_rows,
    pose_reliability_rows,
    source_summary_rows,
)
from components.dm_viz import horizontal_bar_figure, load_json_file
from components.ui import explanation_card, metric_row, render_sidebar
from components.data import load_curated_samples


st.set_page_config(page_title="Dataset Explorer", page_icon="🗂️", layout="wide")

sample_id = render_sidebar()
st.title("Dataset Explorer")

explanation_card(
    "What data supports the pipeline?",
    "This page summarizes the skeleton CSV sources, class balance, curated sample bank, and label/pose-quality metadata. "
    "It helps defend that the demo cases are not isolated screenshots: they are connected to a larger pose dataset and a reviewed curated manifest.",
)

DATA_ROOT = PROJECT_ROOT / "data"
MANIFEST_PATH = PROJECT_ROOT / "webapp" / "artifacts" / "curated" / "manifest.json"


@st.cache_data(show_spinner=False)
def _load_dataset_summaries():
    return dataset_csv_summaries(DATA_ROOT)


summaries = _load_dataset_summaries()
class_rows = class_distribution_rows(summaries)
source_rows = source_summary_rows(summaries)
samples = load_curated_samples()
manifest_payload = load_json_file(MANIFEST_PATH)
curated_rows = curated_sample_rows(samples, selected_sample_id=sample_id)
pose_rows = pose_reliability_rows(manifest_payload)
quality_rows = label_quality_rows(samples)

total_rows = sum(row["rows"] for row in summaries)
total_samples = sum(row["samples"] for row in summaries)
total_files = len(summaries)

st.header("Dataset overview")
metric_row([
    ("Skeleton CSV files", str(total_files), "CSV files discovered under data source folders"),
    ("Unique stroke samples", str(total_samples), "Unique ids summed across source CSVs"),
    ("Frame rows", str(total_rows), "Total per-frame skeleton rows"),
    ("Curated cases", str(len(samples)), "Presentation-ready reviewed cases"),
])

st.caption(
    "Counts are computed from available local CSV artifacts. A CSV row is one frame; unique ids approximate stroke clips/samples."
)

st.divider()
st.header("Class and source distribution")

col_class, col_source = st.columns(2)
with col_class:
    st.subheader("Samples by stroke class")
    if class_rows:
        st.pyplot(horizontal_bar_figure(class_rows, "stroke_type", "samples", "Unique samples by class"))
        st.dataframe(pd.DataFrame(class_rows), width="stretch")
    else:
        st.info("No skeleton CSV summaries found.")

with col_source:
    st.subheader("Samples by source folder")
    if source_rows:
        st.pyplot(horizontal_bar_figure(source_rows, "source", "samples", "Unique samples by source"))
        st.dataframe(pd.DataFrame(source_rows), width="stretch")
    else:
        st.info("No source summaries found.")

with st.expander("Technical: discovered CSV files", expanded=st.session_state.get("detail_level") == "Technical"):
    if summaries:
        st.dataframe(pd.DataFrame(summaries), width="stretch")
    else:
        st.info("No CSV files found in known source folders.")

st.divider()
st.header("Curated sample bank")

curated_cols = st.columns([1, 1])
with curated_cols[0]:
    st.subheader("Curated manifest entries")
    if curated_rows:
        st.dataframe(pd.DataFrame(curated_rows), width="stretch")
    else:
        st.info("No curated samples loaded from manifest.")

with curated_cols[1]:
    st.subheader("Manual review / label trust")
    if quality_rows:
        st.pyplot(horizontal_bar_figure(quality_rows, "manual_review_status", "count", "Curated label review status"))
        st.dataframe(pd.DataFrame(quality_rows), width="stretch")
    else:
        st.info("No manual review metadata found.")

st.divider()
st.header("Pose quality distribution for curated cases")

if pose_rows:
    pose_df = pd.DataFrame(pose_rows)
    metric_row([
        ("Mean pose reliability", f"{pose_df['pose_reliability_score'].mean():.3f}", "Average over curated manifest"),
        ("Min pose reliability", f"{pose_df['pose_reliability_score'].min():.3f}", "Lowest curated pose score"),
        ("Pose-warning cases", str(int((pose_df['warning_count'] > 0).sum())), "Curated cases with at least one warning"),
        ("High reliability cases", str(int((pose_df['pose_reliability_label'] == 'high').sum())), "Cases labelled high reliability"),
    ])
    st.pyplot(horizontal_bar_figure(pose_rows, "sample_id", "pose_reliability_score", "Curated pose reliability by sample"))
    st.dataframe(pose_df, width="stretch")
else:
    st.info("No pose reliability metadata found in manifest.")

st.divider()
st.header("Selected sample context")

selected = next((sample for sample in samples if sample.sample_id == sample_id), None)
if selected:
    st.subheader(selected.title)
    st.write("**Sample ID:**", selected.sample_id)
    st.write("**Stroke type:**", selected.stroke_type)
    st.write("**Ground truth:**", selected.ground_truth or "unknown")
    st.write("**Video path:**", selected.video_path)
    st.write("**Tags:**", ", ".join(selected.tags) or "none")
    st.write("**Teaching point:**", selected.teaching_point)
    st.write("**Diagnosis:**", selected.diagnosis)
else:
    st.info("No selected curated sample available.")

with st.expander("Technical: raw curated manifest", expanded=False):
    st.json(manifest_payload)

st.caption(
    "Dataset Explorer scope: static local dataset and curated-manifest inspection. "
    "Future work can add source-specific train/validation/test split views and richer pose-quality histograms."
)
