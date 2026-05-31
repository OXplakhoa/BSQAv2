# BSQAv2 Pipeline Observatory

Streamlit defense/demo app for the BSQAv2 badminton stroke pipeline.

The Observatory is built around **curated cached artifacts** so the live demo is fast and stable:

```text
video -> MediaPipe/COCO-17 skeleton -> pose QC -> normalized tensor -> RF + DL inference -> Phase 4 quality scoring -> diagnostics
```

## Run

From the repository root:

```bash
cd BSQAv2
../.venv/Scripts/python.exe -m streamlit run webapp/Home.py
```

Use `../.venv/Scripts/python.exe`; plain `python` may point to the Microsoft Store alias on Windows.

## Pages

| Page | Purpose |
|---|---|
| Home | selected curated case summary and navigation map |
| Full Pipeline Demo | end-to-end cached pipeline walkthrough |
| Pose Inspector | raw skeleton, missing joints, visibility, pose warnings |
| Deep Learning Inspector | GCN + BiLSTM + Attention probabilities, shapes, attention maps |
| Data Mining Motion Lab | RF probabilities, biomechanical features, entropy, MI, decision tree, feature importance |
| Error Analysis Lab | RF-vs-DL agreement, confidence margins, pose risk, class-confusion explanations |
| Training & Evaluation | final metrics, per-class F1, confusion matrices, checkpoint inventory |
| Dataset Explorer | source/class distributions, curated manifest, pose-reliability summary |
| Robustness Experiment | artificial skeleton degradation and RF sensitivity analysis |
| Custom Upload | beta live MediaPipe + RF pipeline for a short uploaded video, plus Phase 4 technique-quality feedback |

## Required local artifacts

The curated demo expects these local artifacts:

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
results/dm_analysis/decision_tree_results.json
results/dm_analysis/decision_tree_rules.txt
results/dm_analysis/entropy_analysis.json
results/dm_analysis/mutual_information.csv
results/dm_analysis/*.png
_colab_results/gcn_bilstm_attn_20260528_095136/best_model_fold*.pth
_colab_results/gcn_bilstm_attn_20260528_095136/cv_summary.json
```

Upload mode writes temporary videos here:

```text
webapp/artifacts/uploads/*
```

Most generated artifacts are gitignored but exist in the current local workspace.

If artifacts are missing, regenerate the core set:

```bash
cd BSQAv2
../.venv/Scripts/python.exe src/data/rf_baseline.py --export-artifact
../.venv/Scripts/python.exe tools/scan_curated_candidates.py --max-total 35 --limit-per-class 7 --dl-checkpoint _colab_results/gcn_bilstm_attn_20260528_095136/best_model_fold3.pth
../.venv/Scripts/python.exe tools/build_curated_manifest.py --report webapp/artifacts/curated/<latest_fold3_report>.json --target-count 12
```

## Phase 4 quality scoring

Custom Upload now includes a Phase 4 technique-quality estimate:

- DTW similarity scorer using same-stroke curated references when available,
- curated reference-bank loader from cached PipelineRun artifacts,
- stroke-specific biomechanics rules,
- hybrid 0-100 quality report,
- DTW similarity metric + best reference match,
- interpretable feedback strings.

This is intentionally framed as a heuristic 2D-pose indicator, not expert-labeled coaching truth.

## Test / compile

Current webapp/observatory/quality test slice:

```bash
cd BSQAv2
../.venv/Scripts/python.exe -m unittest tests/test_observatory_artifacts.py tests/test_observatory_pipeline.py tests/test_scan_curated_candidates.py tests/test_observatory_dl_inference.py tests/test_build_curated_manifest.py tests/test_webapp_components.py tests/test_deep_learning_viz.py tests/test_data_mining_viz.py tests/test_error_analysis_viz.py tests/test_eval_viz.py tests/test_dataset_viz.py tests/test_robustness_viz.py tests/test_upload_pipeline.py tests/test_quality_dtw.py tests/test_quality_rules.py tests/test_quality_hybrid.py tests/test_quality_references.py
../.venv/Scripts/python.exe -m compileall src/quality src/observatory webapp tests
```

## Presentation notes

- RF is the strongest numerical classifier in the current artifacts.
- GCN + BiLSTM + Attention is the main Deep Learning architecture and provides spatial-temporal/attention inspection.
- DL should be presented honestly: useful for architecture explanation and attention analysis, but not better than RF numerically here.
- Curated mode is the stable live-defense path.
- Custom upload is available as a beta authenticity path, but it is slower and depends on live MediaPipe extraction.
