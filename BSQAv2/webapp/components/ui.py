"""Reusable Streamlit UI fragments."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import streamlit as st

from .data import default_sample_id, load_curated_samples, sample_label


def render_sidebar() -> Optional[str]:
    st.sidebar.title("BSQAv2 Observatory")
    mode = st.sidebar.radio(
        "Mode",
        ["Curated Investigation", "Custom Upload (coming soon)"],
        index=0,
    )
    st.sidebar.toggle("Guided Presentation Mode", value=True, key="guided_mode")
    st.sidebar.selectbox("Detail level", ["Simple", "Technical"], key="detail_level")

    if mode != "Curated Investigation":
        st.sidebar.info("Custom upload will use the same pipeline after the curated demo is stable.")
        return None

    samples = load_curated_samples()
    if not samples:
        st.sidebar.error("No curated samples found. Build webapp/artifacts/curated/manifest.json first.")
        return None

    sample_by_label = {sample_label(sample): sample.sample_id for sample in samples}
    labels = list(sample_by_label.keys())
    selected_label = st.sidebar.selectbox("Selected curated case", labels)
    return sample_by_label[selected_label]


def explanation_card(title: str, body: str) -> None:
    if st.session_state.get("guided_mode", True):
        st.info(f"**{title}**\n\n{body}")


def metric_row(metrics: List[tuple]) -> None:
    columns = st.columns(len(metrics))
    for column, (label, value, help_text) in zip(columns, metrics):
        column.metric(label, value, help=help_text)


def show_video(video_path: str) -> None:
    path = Path(video_path)
    if path.exists():
        st.video(str(path))
    else:
        st.warning(f"Video not found: {video_path}")
