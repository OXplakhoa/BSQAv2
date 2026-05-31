# BSQAv2 Streamlit Pipeline Observatory — Completion Status

Date: 2026-05-30

## Summary

The Streamlit Pipeline Observatory is complete for both curated defense mode and beta custom upload mode.

The app now demonstrates the full project story:

```text
video -> MediaPipe/COCO-17 skeleton -> pose QC -> preprocessing
      -> Random Forest / Data Mining branch
      -> GCN + BiLSTM + Attention branch
      -> Phase 4 quality scoring: DTW similarity + biomechanics rules
      -> diagnostics, error analysis, evaluation, dataset inspection, robustness
```

## Run command

From repository root:

```bash
cd BSQAv2
../.venv/Scripts/python.exe -m streamlit run webapp/Home.py
```

Use `../.venv/Scripts/python.exe` instead of plain `python` on Windows.

## Completed pages

| Page | Status | Notes |
|---|---:|---|
| Home | ✅ | selected case summary + complete navigation map |
| Full Pipeline Demo | ✅ | cached video-to-prediction pipeline walkthrough |
| Pose Inspector | ✅ | skeleton frame viewer, missing joints, visibility, pose warnings |
| Deep Learning Inspector | ✅ | DL probabilities, architecture path, tensor shapes, attention curve/heatmap |
| Data Mining Motion Lab | ✅ | RF probabilities, features, class averages, entropy, MI, Decision Tree, figures |
| Error Analysis Lab | ✅ | RF/DL agreement, correctness, confidence margins, pose risk, ambiguity cases |
| Training & Evaluation | ✅ | final metrics, per-class F1, confusion matrices, checkpoint inventory |
| Dataset Explorer | ✅ | source/class distribution, curated manifest, pose-reliability summary |
| Robustness Experiment | ✅ | artificial skeleton degradation + RF sensitivity curve |
| Custom Upload | ✅ beta | live MediaPipe + RF path, Phase 4 technique-quality feedback; optional DL path verified but slow |

## Smoke verification summary

All pages were manually smoke-tested by the user's subagent.

Verified properties across pages:

- sidebar navigation works,
- curated sample selector works,
- videos render,
- charts/images/dataframes render,
- Technical mode expanders work,
- no Streamlit tracebacks,
- no Matplotlib exceptions,
- only benign Streamlit health-check 404s remain in browser console.

Custom Upload was live-tested with:

```text
BSQAv2/data/clips/smash/iuuLXZ4g8bc_046.mp4
```

Verified upload flow:

```text
Upload MP4 -> save_uploaded_file() -> extract_keypoints_from_video()
  -> MediaPipe Pose -> COCO-17 keypoints -> pose QC -> preprocessing
  -> extract_features() -> rf_bundle.predict() -> PipelineRun
  -> optional DL checkpoint inference
```

The ground-truth persistence bug was fixed by wrapping upload controls in `st.form()`.

## Phase 4 quality assessment

Phase 4 has now been implemented as a heuristic technique-quality layer separate from pose reliability.

Implemented files:

```text
src/quality/dtw_scorer.py
src/quality/rules.py
src/quality/hybrid.py
src/observatory/quality_references.py
tests/test_quality_dtw.py
tests/test_quality_rules.py
tests/test_quality_hybrid.py
tests/test_quality_references.py
docs/PHASE4_QUALITY_ASSESSMENT_PLAN.md
```

Behavior:

- DTW compares `(T, 17, 2)` skeletons against same-stroke references.
- Biomechanics rules return stroke-specific 0-100 checks and feedback for smash, clear, drop_shot, net_shot, lift.
- Hybrid scorer combines DTW and rules with default weights: DTW 40%, rules 60%.
- If references are unavailable, the system returns a valid rule-only quality report.
- `src/observatory/quality_references.py` loads a small per-stroke reference bank from curated cached PipelineRun artifacts.
- Current local reference bank: smash=2, clear=3, drop_shot=1, net_shot=3, lift=3.
- `run_skeleton_pipeline()` now accepts a same-stroke reference list or stroke->references dict, then adds `diagnostics["quality_report"]` and `diagnostics["quality_summary"]`.
- Custom Upload passes the curated reference bank, shows a `Technique quality` metric, a `DTW similarity` metric when available, best reference match, and interpretable rule feedback.

Honest framing:

- These are heuristic 2D-pose indicators, not expert-supervised coaching labels.
- Scores should be described as technique-similarity / biomechanical indicators.
- DTW quality depends on reference-sample quality; current upload path uses the curated reference bank when same-stroke references exist.

## Automated tests

Current observatory/webapp/quality test command:

```bash
cd BSQAv2
../.venv/Scripts/python.exe -m unittest \
  tests/test_observatory_artifacts.py \
  tests/test_observatory_pipeline.py \
  tests/test_scan_curated_candidates.py \
  tests/test_observatory_dl_inference.py \
  tests/test_build_curated_manifest.py \
  tests/test_webapp_components.py \
  tests/test_deep_learning_viz.py \
  tests/test_data_mining_viz.py \
  tests/test_error_analysis_viz.py \
  tests/test_eval_viz.py \
  tests/test_dataset_viz.py \
  tests/test_robustness_viz.py \
  tests/test_upload_pipeline.py \
  tests/test_quality_dtw.py \
  tests/test_quality_rules.py \
  tests/test_quality_hybrid.py \
  tests/test_quality_references.py
```

Latest result:

```text
Ran 69 tests
OK
```

Compile command:

```bash
cd BSQAv2
../.venv/Scripts/python.exe -m compileall src/quality src/observatory webapp tests
```

Latest result: no compile errors.

## Key implementation files

### Observatory backend

```text
src/observatory/schema.py
src/observatory/artifacts.py
src/observatory/dm_inference.py
src/observatory/dl_inference.py
src/observatory/diagnostics.py
src/observatory/pose_quality.py
src/observatory/pipeline.py
src/observatory/quality_references.py
src/observatory/upload_pipeline.py
```

### Webapp pages

```text
webapp/Home.py
webapp/pages/1_Full_Pipeline_Demo.py
webapp/pages/2_Pose_Inspector.py
webapp/pages/3_Deep_Learning_Inspector.py
webapp/pages/4_Data_Mining_Motion_Lab.py
webapp/pages/5_Error_Analysis_Lab.py
webapp/pages/6_Training_and_Evaluation.py
webapp/pages/7_Dataset_Explorer.py
webapp/pages/8_Robustness_Experiment.py
webapp/pages/9_Custom_Upload.py
```

### Webapp components

```text
webapp/components/bootstrap.py
webapp/components/charts.py
webapp/components/data.py
webapp/components/dataset_viz.py
webapp/components/dl_viz.py
webapp/components/dm_viz.py
webapp/components/error_viz.py
webapp/components/eval_viz.py
webapp/components/robustness_viz.py
webapp/components/skeleton_view.py
webapp/components/ui.py
```

### Tools

```text
tools/scan_curated_candidates.py
tools/build_curated_manifest.py
```

## Important local artifacts

Curated mode depends on local generated artifacts:

```text
webapp/artifacts/curated/manifest.json
webapp/artifacts/pipeline_runs/*/run.json
webapp/artifacts/pipeline_runs/*/arrays/*.npy
webapp/artifacts/models/rf_baseline/rf_model_bundle.joblib
webapp/artifacts/models/rf_baseline/rf_model_manifest.json
results/rf_baseline/rf_results.json
results/rf_baseline/rf_confusion_matrix.png
results/rf_baseline/rf_confusion_matrix_norm.png
results/rf_baseline/rf_feature_importance.png
results/dm_analysis/*
_colab_results/gcn_bilstm_attn_20260528_095136/best_model_fold*.pth
_colab_results/gcn_bilstm_attn_20260528_095136/cv_summary.json
```

Generated upload videos are ignored by git:

```text
webapp/artifacts/uploads/**
```

## Model/evaluation story to present

- Random Forest is the strongest numerical classifier in the current artifacts.
- Final DL metrics are shown from the PRD/report values:
  - Accuracy: `0.6563 ± 0.0372`
  - F1 macro: `0.6483 ± 0.0408`
  - F1 weighted: `0.6517 ± 0.0400`
- RF metrics:
  - Accuracy: `0.7172`
  - F1 macro: `0.7134`
  - F1 weighted: `0.7161`
- The DL model remains important as the proposed spatial-temporal architecture:
  - Spatial GCN,
  - BiLSTM,
  - Temporal Attention,
  - attention-based interpretability.
- Lift is explicitly presented as a weak/difficult DL class.

## Known limitations / honest framing

- Custom Upload is beta and slower because MediaPipe runs frame-by-frame.
- Optional DL upload inference is functional but slow on CPU; RF-only upload is the practical live path.
- The app should not claim correctness for uploads unless the user supplies ground truth.
- Skeleton-only models lack shuttle/racket/court context.
- Phase 4 quality scoring is heuristic because the project has no expert quality labels.
- RF outperforming DL should be presented honestly as a data-size/noise/engineered-feature advantage.

## Git note

Many files are untracked because this repository had broad untracked project work already. Do not treat untracked Observatory files as disposable.

Notable areas expected to be added/committed together:

```text
BSQAv2/.gitignore
BSQAv2/docs/STREAMLIT_PIPELINE_OBSERVATORY_PRD.md
BSQAv2/docs/STREAMLIT_PIPELINE_OBSERVATORY_COMPLETION.md
BSQAv2/docs/PHASE4_QUALITY_ASSESSMENT_PLAN.md
BSQAv2/src/observatory/
BSQAv2/src/quality/
BSQAv2/tests/
BSQAv2/tools/scan_curated_candidates.py
BSQAv2/tools/build_curated_manifest.py
BSQAv2/webapp/
```
