from components.bootstrap import PROJECT_ROOT, ensure_project_imports
ensure_project_imports()

import pandas as pd
import streamlit as st

from components.dm_viz import existing_path, horizontal_bar_figure, load_json_file
from components.eval_viz import (
    QUALITY_METRIC_NOTE,
    available_training_curve_folds,
    available_training_curve_tags,
    comparison_rows,
    discover_training_run_dirs,
    dl_final_metrics,
    fold_artifact_rows,
    load_tensorboard_scalars,
    metric_card_value,
    per_class_metric_rows,
    rf_summary_metrics,
    training_curve_figure,
)
from components.ui import explanation_card, metric_row, render_glossary_footer, render_sidebar
from src.evaluation.manual_quality import load_manual_quality_results, summarize_manual_quality_results


st.set_page_config(page_title="Training & Evaluation", page_icon="📊", layout="wide")

sample_id = render_sidebar()
st.title("Training & Evaluation — Huấn luyện & đánh giá")

explanation_card(
    "So sánh các mô hình cuối cùng",
    "Trang này tách câu chuyện evaluation học thuật khỏi demo một mẫu. "
    "Random Forest là classifier có số liệu mạnh nhất trong artifact hiện tại, còn "
    "GCN + BiLSTM + Attention vẫn là kiến trúc Deep Learning chính vì nó mô hình hóa "
    "skeleton graph, chuyển động theo thời gian, và frame importance dựa trên attention.",
)

RESULTS_ROOT = PROJECT_ROOT / "results"
COLAB_RUN_DIR = PROJECT_ROOT / "_colab_results" / "gcn_bilstm_attn_20260528_095136"
RF_RESULTS_PATH = RESULTS_ROOT / "rf_baseline" / "rf_results.json"
TREE_RESULTS_PATH = RESULTS_ROOT / "dm_analysis" / "decision_tree_results.json"
CV_SUMMARY_PATH = COLAB_RUN_DIR / "cv_summary.json"
MANUAL_QUALITY_RESULTS_PATH = PROJECT_ROOT / "data" / "manual_quality_evaluation_50.csv"

rf_results = load_json_file(RF_RESULTS_PATH)
tree_results = load_json_file(TREE_RESULTS_PATH)
local_cv_summary = load_json_file(CV_SUMMARY_PATH)
dl_metrics = dl_final_metrics()
rf_metrics = rf_summary_metrics(rf_results)
comparison = comparison_rows(rf_results, tree_results, dl_metrics=dl_metrics)
per_class_rows = per_class_metric_rows(rf_results, dl_metrics=dl_metrics)
checkpoint_paths = sorted(COLAB_RUN_DIR.glob("best_model_fold*.pth")) if COLAB_RUN_DIR.exists() else []
event_paths = sorted(COLAB_RUN_DIR.glob("fold_*/events.out.tfevents.*")) if COLAB_RUN_DIR.exists() else []
manual_quality_rows = load_manual_quality_results(MANUAL_QUALITY_RESULTS_PATH)
manual_quality_summary = summarize_manual_quality_results(manual_quality_rows)

st.header("Metrics chính cuối cùng")
metric_row([
    ("DL accuracy", f"{dl_metrics['accuracy']:.3f} ± {dl_metrics['accuracy_std']:.3f}", "Final GCN + BiLSTM + Attention CV result from report/PRD"),
    ("DL F1 macro", f"{dl_metrics['f1_macro']:.3f} ± {dl_metrics['f1_macro_std']:.3f}", "Macro F1 across five stroke classes"),
    ("RF accuracy", metric_card_value(rf_metrics.get("accuracy")), "Random Forest baseline artifact"),
    ("RF F1 macro", metric_card_value(rf_metrics.get("f1_macro")), "Random Forest macro F1"),
])

st.info(
    "Main conclusion: thêm các ablation LSTM / BiLSTM / GCN+LSTM vào bảng so sánh. "
    "RF vẫn là non-DL baseline mạnh; GCN + BiLSTM + Attention vẫn là kiến trúc spatial-temporal chính vì có Attention inspection."
)

ablation_rows = [row for row in comparison if row.get("model") in {"LSTM", "BiLSTM", "GCN + LSTM"}]
if ablation_rows:
    st.subheader("Kết quả 3 model ablation cần bổ sung vào báo cáo")
    st.dataframe(pd.DataFrame(ablation_rows), width="stretch")

st.divider()
st.header("So sánh mô hình")

comparison_df = pd.DataFrame(comparison)
if not comparison_df.empty:
    st.dataframe(comparison_df, width="stretch")
    chart_rows = [
        {"model": row["model"], "accuracy": row["accuracy"]}
        for row in comparison
        if row.get("accuracy") is not None
    ]
    st.pyplot(horizontal_bar_figure(chart_rows, "model", "accuracy", "Accuracy by model"))
    speed_rows = [
        {"model": row["model"], "inference_ms_per_frame": row["inference_ms_per_frame"]}
        for row in comparison
        if row.get("inference_ms_per_frame") is not None
    ]
    if speed_rows:
        st.pyplot(horizontal_bar_figure(speed_rows, "model", "inference_ms_per_frame", "Inference speed (ms/frame, lower is faster)"))
    st.caption(QUALITY_METRIC_NOTE)
else:
    st.warning("Không tìm thấy artifact so sánh mô hình.")

st.divider()
st.header("Manual quality subset — MAE / Spearman rs")

mq_summary = manual_quality_summary.get("summary", {})
if mq_summary.get("n_samples", 0):
    metric_row([
        ("Quality subset samples", str(mq_summary.get("n_samples")), "Manual 0-100 labels, 10 clips per stroke class"),
        ("Quality MAE", metric_card_value(mq_summary.get("mae")), "Mean absolute error: Phase 4 scorer vs manual labels"),
        ("Spearman rs", metric_card_value(mq_summary.get("spearman_rs")), "Rank correlation: Phase 4 scorer vs manual labels"),
        ("Manual mean", metric_card_value(mq_summary.get("manual_mean")), "Average manual quality score"),
    ])
    st.dataframe(pd.DataFrame(manual_quality_summary.get("per_stroke", [])), width="stretch")
    st.caption(
        "Honest scope: đây là evaluation của Phase 4 heuristic quality scorer, không phải quality head được train supervised. "
        "50 label này giúp báo cáo có MAE/Spearman thật trên subset thủ công."
    )
else:
    st.info("Chưa có manual quality subset result.")

st.divider()
st.header("Hiệu năng theo từng class")

if per_class_rows:
    per_class_df = pd.DataFrame(per_class_rows)
    st.dataframe(per_class_df, width="stretch")

    dl_chart_rows = [
        {"class": row["class"], "dl_f1": row.get("dl_f1") or 0.0}
        for row in per_class_rows
    ]
    rf_chart_rows = [
        {"class": row["class"], "rf_f1": row.get("rf_f1") or 0.0}
        for row in per_class_rows
    ]
    col_rf, col_dl = st.columns(2)
    with col_rf:
        st.pyplot(horizontal_bar_figure(rf_chart_rows, "class", "rf_f1", "RF per-class F1"))
    with col_dl:
        st.pyplot(horizontal_bar_figure(dl_chart_rows, "class", "dl_f1", "DL per-class F1"))
    st.warning(
        "Lift is the weakest DL class (F1 ≈ 0.434), so lift/clear and lift/net-shot cases are useful error-analysis examples."
    )
else:
    st.info("Không có per-class metrics.")

st.divider()
st.header("Artifact đánh giá")

artifact_cols = st.columns(3)
with artifact_cols[0]:
    st.subheader("RF confusion matrix")
    path = existing_path(RESULTS_ROOT / "rf_baseline" / "rf_confusion_matrix.png")
    if path:
        st.image(str(path), caption="RF confusion matrix")
    else:
        st.info("Thiếu ảnh RF confusion matrix.")

with artifact_cols[1]:
    st.subheader("RF confusion matrix đã normalize")
    path = existing_path(RESULTS_ROOT / "rf_baseline" / "rf_confusion_matrix_norm.png")
    if path:
        st.image(str(path), caption="RF confusion matrix đã normalize")
    else:
        st.info("Thiếu ảnh normalized RF confusion matrix.")

with artifact_cols[2]:
    st.subheader("RF feature importance")
    path = existing_path(RESULTS_ROOT / "rf_baseline" / "rf_feature_importance.png")
    if path:
        st.image(str(path), caption="RF feature importance")
    else:
        st.info("Thiếu ảnh RF feature importance.")

st.divider()
st.header("Danh sách training run")

fold_rows = fold_artifact_rows(local_cv_summary, [path.name for path in checkpoint_paths]) if local_cv_summary else []
if fold_rows:
    st.dataframe(pd.DataFrame(fold_rows), width="stretch")
else:
    st.info("Không tìm thấy local DL CV summary.")

metric_row([
    ("Checkpoints", str(len(checkpoint_paths)), "best_model_fold*.pth files found locally"),
    ("TensorBoard event files", str(len(event_paths)), "Training event logs found under fold_* directories"),
    ("Local run CV accuracy", metric_card_value(local_cv_summary.get("accuracy_mean") if local_cv_summary else None), "Exploratory local/Colab run summary"),
    ("Local run F1 macro", metric_card_value(local_cv_summary.get("f1_macro_mean") if local_cv_summary else None), "Exploratory local/Colab run summary"),
])

if local_cv_summary:
    st.caption(
        "Note: the checkpoint folder used for demo inference may have exploratory metrics that differ from the final report/PRD values. "
        "The headline DL metrics above use the final selected evaluation values."
    )

st.divider()
st.header("Training curves — train/val loss & accuracy")
st.caption(
    "Đọc trực tiếp từ TensorBoard scalar logs do `train.py` ghi: "
    "Loss/train, Loss/val, Acc/train, Acc/val, F1/val_macro, LR."
)

RUNS_ROOT = PROJECT_ROOT / "runs"
training_run_dirs = discover_training_run_dirs(COLAB_RUN_DIR, RUNS_ROOT)
if training_run_dirs:
    default_idx = 0
    for idx, path in enumerate(training_run_dirs):
        if path == COLAB_RUN_DIR:
            default_idx = idx
            break
    selected_run = st.selectbox(
        "Training run directory",
        training_run_dirs,
        index=default_idx,
        format_func=lambda path: path.relative_to(PROJECT_ROOT).as_posix() if path.is_relative_to(PROJECT_ROOT) else str(path),
    )

    @st.cache_data(show_spinner=False)
    def _load_curves(path_text: str):
        return load_tensorboard_scalars(Path(path_text))

    curve_rows = _load_curves(str(selected_run))
    event_files_for_run = sorted(selected_run.glob("fold_*/events.out.tfevents.*"))
    if curve_rows:
        tags = available_training_curve_tags(curve_rows)
        folds = ["All"] + available_training_curve_folds(curve_rows)
        selected_fold = st.selectbox("Fold", folds)
        selected_tags = st.multiselect(
            "Curves",
            tags,
            default=[tag for tag in ["Loss/train", "Loss/val", "Acc/train", "Acc/val"] if tag in tags] or tags[:2],
        )
        if selected_tags:
            st.pyplot(training_curve_figure(curve_rows, selected_tags=selected_tags, selected_fold=selected_fold, title=f"Training curves — {selected_run.name}"))
        else:
            st.info("Chọn ít nhất một curve để hiển thị.")
        if st.session_state.get("detail_level") == "Technical":
            st.dataframe(pd.DataFrame(curve_rows), width="stretch")
    else:
        st.warning(
            "Có TensorBoard event files, nhưng không đọc được scalar curves từ run này. "
            "Điều này thường xảy ra nếu event file được copy chỉ chứa metadata, logs chưa flush, hoặc run cũ không ghi scalar."
        )
        st.write(f"Event files found: `{len(event_files_for_run)}`")
        st.code(
            f"../.venv/Scripts/python.exe -m tensorboard.main --logdir {selected_run.relative_to(PROJECT_ROOT).as_posix() if selected_run.is_relative_to(PROJECT_ROOT) else selected_run}",
            language="bash",
        )
        st.info(
            "Muốn có curve thật, train lại bằng `train.py`; training loop hiện đã log "
            "Loss/train, Loss/val, Acc/train, Acc/val, F1/val_macro và LR qua TensorBoard SummaryWriter."
        )
else:
    st.info("Không tìm thấy TensorBoard event files dưới `runs/` hoặc Colab checkpoint folder.")

with st.expander("Technical: raw metric artifacts", expanded=st.session_state.get("detail_level") == "Technical"):
    st.write("**RF results JSON**")
    st.json(rf_results)
    st.write("**Decision Tree results JSON**")
    st.json(tree_results)
    st.write("**Local DL cv_summary.json**")
    st.json(local_cv_summary)

st.caption(
    "Evaluation dashboard scope: report-ready comparison of RF, Decision Tree, LSTM, BiLSTM, GCN+LSTM, "
    "final GCN + BiLSTM + Attention, and TensorBoard training-curve inspection when scalar logs are available."
)

render_glossary_footer()
