# PRD: BSQAv2 Pipeline Observatory Streamlit Demo

## Problem Statement

BSQAv2 is a university project for both Data Mining and Deep Learning. The system is not a simple classifier: it is a multi-stage human-motion analysis pipeline where raw badminton videos pass through MediaPipe Pose, skeleton conversion, preprocessing, and then either a deep learning model or a data mining model.

A simple prediction demo would not be enough. The lecturers and viewers need to understand:

- how a raw video becomes skeleton data,
- how MediaPipe pose quality affects downstream prediction,
- how the GCN + BiLSTM + Attention model processes a skeleton sequence,
- how Data Mining extracts interpretable biomechanical features,
- how Random Forest and Decision Tree models relate to entropy, mutual information, and feature discovery,
- why a prediction may be wrong,
- how training loss, accuracy, F1, confusion matrices, and per-class performance changed across experiments,
- why this problem is complex and why errors can come from different stages of the pipeline.

The current project already has valuable assets: manually reviewed premium clips in `data/clips`, trained DL checkpoints, Random Forest results, Data Mining analysis figures, entropy analysis, decision tree rules, feature importance charts, and final CV results. However, these assets are spread across scripts, results folders, and Colab outputs. The project needs a robust, visually rich, audience-aware Streamlit app that turns these assets into a coherent demo and defense tool.

## Solution

Build a multi-page Streamlit app called **BSQAv2 Pipeline Observatory**.

The app will demonstrate the complete pipeline:

1. Input badminton stroke video.
2. Run MediaPipe Pose.
3. Convert MediaPipe landmarks to COCO-17 skeleton keypoints.
4. Run pose quality diagnostics.
5. Preprocess skeletons into normalized 64-frame sequences.
6. Run the Deep Learning branch using GCN + BiLSTM + Temporal Attention.
7. Run the Data Mining branch using biomechanical features + Random Forest.
8. Visualize model predictions, confidence, attention, features, graphs, and diagnostics.
9. Explain correct and incorrect predictions through pose quality, confidence margin, class confusion, feature evidence, and RF vs DL disagreement.
10. Provide training and evaluation dashboards for accuracy, loss, F1, confusion matrices, ablation comparisons, and robustness analysis.

The app will support two interaction modes:

- **Curated Investigation Mode**: uses 12 curated samples from the user's manually reviewed `data/clips` dataset. This mode is optimized for reliable presentation and explanation.
- **Custom Video Upload Mode**: allows users to upload their own badminton video and run the real MediaPipe + DL + DM pipeline. Ground truth is optional. If no ground truth is provided, the app will show prediction, confidence, and pose reliability, but will not claim correctness.

The app will also support two audience detail levels:

- **Simple View**: plain-language explanation, main graphs, key conclusions, minimal jargon.
- **Technical View**: tensor shapes, model internals, feature values, thresholds, attention matrices, metrics, training curves, and detailed diagnostics.

A **Guided Presentation Mode** will add explanation cards to each stage: what it does, what to look at, and why it matters.

## User Stories

1. As a Deep Learning lecturer, I want to see the full video-to-model pipeline, so that I can verify the project is not only classifying precomputed CSV files.
2. As a Deep Learning lecturer, I want to see MediaPipe Pose running before the model, so that I can understand the two-layer architecture of the system.
3. As a Deep Learning lecturer, I want to see the conversion from MediaPipe landmarks to COCO-17 keypoints, so that I can verify the pose representation used by the model.
4. As a Deep Learning lecturer, I want to see the normalized 64-frame skeleton tensor shape, so that I can understand the model input.
5. As a Deep Learning lecturer, I want to see the GCN, BiLSTM, and Attention stages separately, so that I can understand the architecture contribution.
6. As a Deep Learning lecturer, I want to see class probabilities, so that I can judge model confidence rather than only the final label.
7. As a Deep Learning lecturer, I want to see the attention heatmap and frame importance curve, so that I can inspect which parts of the stroke the model focused on.
8. As a Deep Learning lecturer, I want to see training loss and validation metrics, so that I can evaluate whether the model learned properly.
9. As a Deep Learning lecturer, I want to see confusion matrices and per-class F1 scores, so that I can understand which strokes are hard for the model.
10. As a Deep Learning lecturer, I want to see the LSTM baseline in the evaluation dashboard, so that I can compare the proposed DL architecture against a baseline.
11. As a Data Mining lecturer, I want to see skeletons converted into biomechanical features, so that I can understand how the raw motion data becomes a mining dataset.
12. As a Data Mining lecturer, I want to see entropy and class distribution, so that I can connect the demo to Data Mining theory.
13. As a Data Mining lecturer, I want to see mutual information rankings, so that I can understand which features reduce uncertainty about the stroke class.
14. As a Data Mining lecturer, I want to see feature distribution plots by class, so that I can visually compare biomechanical patterns between strokes.
15. As a Data Mining lecturer, I want to see feature correlation heatmaps, so that I can understand redundancy between extracted features.
16. As a Data Mining lecturer, I want to see a decision tree visualization, so that I can inspect interpretable rules even if the tree is not the strongest model.
17. As a Data Mining lecturer, I want to see Random Forest feature importance, so that I can understand which biomechanical features drive the strongest DM model.
18. As a Data Mining lecturer, I want to see t-SNE or UMAP feature-space plots, so that I can visually inspect class overlap and clustering.
19. As a Data Mining lecturer, I want to see local explanations for one selected clip, so that global mining results connect to an individual prediction.
20. As a Data Mining lecturer, I want to see current-sample feature values compared with class averages, so that prediction reasoning becomes understandable.
21. As a presenter, I want curated samples with known teaching purposes, so that I can reliably demonstrate both success cases and error cases.
22. As a presenter, I want curated clips to come from the manually reviewed `data/clips` dataset, so that the demo uses premium, visually understandable examples.
23. As a presenter, I want the app to default to curated investigation mode, so that the live presentation is stable and fast.
24. As a presenter, I want to upload a custom video, so that I can prove the real pipeline works beyond cached examples.
25. As a presenter, I want uploaded videos to show pose reliability warnings, so that viewers understand prediction limitations.
26. As a presenter, I want optional ground-truth labels for uploaded videos, so that correctness is only computed when a label is provided.
27. As a presenter, I want fallback wording for unknown labels, so that the app remains academically honest.
28. As a presenter, I want the fallback wording to say the predicted class, confidence, pose reliability, and caution, so that uncertain predictions are responsibly communicated.
29. As a viewer, I want simple explanations, so that I can understand the system even without machine learning background.
30. As a lecturer, I want technical details, so that I can inspect model shapes, metrics, thresholds, and data transformations.
31. As a viewer, I want skeleton overlays, so that I can visually understand what the model sees.
32. As a viewer, I want graphs organized by pipeline stage, so that the demo feels clear rather than cluttered.
33. As a viewer, I want only a few main graphs per page, so that the interface is understandable.
34. As a technical viewer, I want extra graphs in expandable sections, so that I can inspect more detail when needed.
35. As a presenter, I want Guided Presentation Mode, so that each stage explains what it does, what to look at, and why it matters.
36. As a presenter, I want Simple View as the default, so that the app is accessible to everyone.
37. As a presenter, I want Technical View available, so that I can answer lecturer questions during defense.
38. As a user, I want one selected sample to control all pages, so that the same clip can be investigated through DL, DM, pose, and error analysis.
39. As a user, I want global DM graphs to highlight the selected sample when possible, so that the sample is connected to the full dataset.
40. As a user, I want an Error Analysis Lab, so that wrong predictions become explainable instead of simply looking like failures.
41. As a user, I want RF vs DL agreement/disagreement analysis, so that I can understand when engineered features help and when temporal DL helps.
42. As a user, I want confidence margin analysis, so that I can distinguish high-confidence predictions from ambiguous ones.
43. As a user, I want pose reliability analysis, so that I can understand when MediaPipe quality may affect prediction.
44. As a user, I want known class-confusion explanations, so that lift-clear and other ambiguous cases are understandable.
45. As a user, I want a robustness experiment, so that I can see how noise, joint dropout, or frame dropout changes predictions.
46. As a user, I want training dashboards, so that I can see how accuracy, F1, and loss changed across experiments.
47. As a user, I want run comparison tables, so that I can understand why training stopped and why the final models were chosen.
48. As a user, I want a dataset explorer, so that I can inspect class distribution, source distribution, and sample-level data quality.
49. As a developer, I want all curated samples cached as artifacts, so that the demo loads quickly and reliably.
50. As a developer, I want custom upload mode to use the same canonical pipeline as curated mode, so that predictions are consistent.
51. As a developer, I want a PipelineRun artifact, so that every page reads from one stable result object.
52. As a developer, I want raw pixel skeletons and normalized skeletons both stored, so that pose diagnostics and model inference can use the correct representation.
53. As a developer, I want the RF model saved as a loadable artifact, so that the app does not retrain Random Forest at startup.
54. As a developer, I want feature names, label encoder, class averages, and feature medians saved, so that RF inference and explanations are reproducible.
55. As a developer, I want t-SNE or UMAP coordinates precomputed, so that the Data Mining feature-space graph loads quickly.
56. As a developer, I want curated sample metadata stored in a manifest, so that each case has a title, source path, ground truth, teaching point, and diagnosis.
57. As a developer, I want cached arrays saved separately from JSON metadata, so that artifacts remain manageable.
58. As a developer, I want a multi-page Streamlit architecture, so that the app remains maintainable and each page has one clear purpose.
59. As a developer, I want reusable UI components for charts, cards, explanations, skeleton viewers, and diagnosis panels, so that pages stay clean.
60. As a developer, I want robust handling of missing models or artifacts, so that the app gives helpful messages instead of crashing.

## Implementation Decisions

- The app will be a multi-page Streamlit app, not one huge page with tabs.
- The app identity is **BSQAv2 Pipeline Observatory**.
- The app will be organized around the pipeline and investigation workflow, not just prediction.
- The app will support **Curated Investigation Mode** and **Custom Video Upload Mode**.
- Curated Investigation Mode is the default and is optimized for reliable presentation.
- Custom Video Upload Mode will run the real MediaPipe + preprocessing + DL + DM pipeline.
- Real-time webcam prediction is out of scope for the main implementation.
- Curated samples will be selected only from the manually reviewed `data/clips` dataset.
- The final curated case bank will contain approximately 12 samples.
- Curated sample selection will happen in two stages: candidate scan, then final curation.
- The candidate scan will evaluate around 30 to 35 clips from `data/clips`.
- Final curated samples will be chosen for visual clarity, pose quality, class coverage, correct cases, error cases, and teaching value.
- The curated case bank should include clean correct cases, model-confusion cases, and pose-warning or low-confidence cases.
- One selected sample will control the whole app state across pages.
- Global dataset graphs may show all samples but should highlight the selected sample when possible.
- The app will use one canonical pipeline for curated samples, uploaded videos, DL inference, DM inference, and diagnosis.
- Each run will produce or load a **PipelineRun** artifact.
- PipelineRun will store video metadata, raw keypoints, visibilities, pose QC metrics, normalized keypoints, DL prediction, DL probabilities, attention weights, DM features, RF prediction, RF probabilities, and diagnosis output.
- Large arrays will be stored separately from JSON metadata.
- Raw pixel skeletons will be used for pose diagnostics.
- Normalized 64-frame skeletons will be used for DL inference.
- DM biomechanical features will be extracted from normalized 64-frame skeletons for consistency with the existing RF training pipeline.
- The app will store both raw pixel keypoints and normalized keypoints in artifacts.
- The main Deep Learning model in live demo pages will be GCN + BiLSTM + Temporal Attention.
- The LSTM model will appear as a baseline in the Training and Evaluation dashboard only.
- The main Data Mining model will be Random Forest.
- Decision Tree will be used for interpretability and rule visualization, not as the main predictor.
- The app will not claim that the GCN + BiLSTM + Attention model is the best numerical model.
- The app will present the DL model as the main spatial-temporal end-to-end architecture.
- The app will present Random Forest as the strongest interpretable Data Mining classifier.
- The app will explicitly explain that engineered biomechanical features can outperform raw-coordinate DL on small/noisy pose datasets.
- The app will include a Label and Data Quality Panel.
- For curated samples, the app will show ground truth, source path, manual review status, pose QC status, and correctness.
- For uploaded samples, ground truth will be optional.
- If no ground truth is provided, the app will not mark predictions as correct or wrong.
- For unknown labels, the app will use cautious fallback wording such as: "Predicted as clear with 43% confidence. Pose reliability is medium. This should be interpreted cautiously."
- Graphs will be organized by pipeline stage.
- Each page will show approximately 3 to 5 primary graphs.
- Additional graphs will be hidden in expandable sections.
- The app will include Guided Presentation Mode.
- Guided Presentation Mode will explain what each stage does, what to look at, and why it matters.
- The app will include Simple View and Technical View.
- Simple View will be the default.
- Technical View will expose tensor shapes, thresholds, detailed metrics, model internals, and raw feature values.
- The app will include an artifact layer containing models, metrics, figures, curated sample cases, and feature-space data.
- The Random Forest model must be saved as a loadable artifact.
- The RF label encoder, feature names, class averages, feature medians, and feature imputers or NaN handling strategy must be saved.
- Precomputed t-SNE or UMAP feature-space coordinates should be generated for fast Data Mining visualization.
- Existing figures from RF baseline and DM analysis should be reused where appropriate.
- Final DL v5 metrics will be shown in the evaluation dashboard.
- The final GCN + BiLSTM + Attention metrics are: accuracy 0.6563 +/- 0.0372, F1 macro 0.6483 +/- 0.0408, F1 weighted 0.6517 +/- 0.0400.
- Final GCN + BiLSTM + Attention per-class F1 values are: smash 0.824, clear 0.644, drop_shot 0.690, net_shot 0.758, lift 0.434.
- RF baseline metrics are: accuracy 0.7172, F1 macro 0.7134, F1 weighted 0.7161.
- The app will emphasize that lift is a difficult class, especially for the DL model.

## Artifact Foundation Contract

The first implementation slice creates the artifact layer before heavy UI work. The artifact layer is the stable interface between slow/fragile pipeline computation and fast Streamlit pages.

Default artifact root:

```text
BSQAv2/webapp/artifacts/
  README.md
  curated/manifest.json
  pipeline_runs/
  models/
    rf_baseline/rf_model_bundle.joblib
    rf_baseline/rf_model_manifest.json
  metrics/
  figures/
  feature_space/
```

Core implementation modules:

- `src/observatory/schema.py`
  - `PipelineRun`
  - `PredictionResult`
  - `CuratedSample`
  - PipelineRun JSON + `.npy` save/load helpers
- `src/observatory/artifacts.py`
  - `ArtifactRegistry`
  - directory bootstrap
  - curated manifest loading
  - PipelineRun load/save resolution
  - RF bundle path validation
- `src/observatory/dm_inference.py`
  - RF bundle loading
  - feature-order validation
  - median imputation during inference
  - RF prediction and class-average comparison helpers
- `src/observatory/diagnostics.py`
  - safe prediction wording
  - RF/DL agreement text
  - confidence margin and pose reliability wording

### PipelineRun v1 Required Contract

A cached PipelineRun directory contains:

```text
pipeline_runs/{run_id}/
  run.json
  arrays/
    raw_keypoints.npy              # shape (T, 17, 2), pixel coordinates
    normalized_keypoints.npy       # shape (64, 17, 2), model coordinates
    visibilities.npy               # optional shape (T, 17)
    attention_weights.npy          # optional shape (64, 64) or model-specific
```

`run.json` stores JSON-safe metadata only:

- run identity: `run_id`, `sample_id`, `mode`, `created_at`
- video metadata and optional ground truth
- pose QC metrics and warning messages
- DL prediction label, probabilities, confidence, attention artifact path
- RF prediction label, probabilities, confidence
- DM feature dictionary
- diagnostics and timing information
- relative paths to large arrays

This split is required because JSON should remain inspectable while skeleton arrays, attention arrays, and frame-level diagnostics stay efficient and lossless.

### Random Forest Artifact Contract

The RF artifact must be generated once and loaded by the app, never retrained at Streamlit startup.

Generation command:

```bash
cd BSQAv2
../.venv/Scripts/python.exe src/data/rf_baseline.py --export-artifact
```

The RF bundle contains:

- fitted `RandomForestClassifier`,
- fitted `LabelEncoder`,
- exact `feature_names` order used during training,
- `feature_medians` for inference-time NaN handling,
- per-class feature averages for local explanations,
- metadata describing training parameters and NaN strategy.

Training still drops NaN rows for CV metrics, matching the existing baseline. Streamlit inference uses saved medians so one missing feature in an uploaded or curated clip does not crash the demo.

### Artifact Foundation Acceptance Criteria

- Artifact directories exist and are documented.
- Empty curated manifest loads without crashing.
- PipelineRun save/load round-trips metadata and arrays.
- Normalized skeleton arrays are validated as `(64, 17, 2)`.
- Raw skeleton arrays are stored separately from normalized skeleton arrays.
- RF export writes a joblib bundle plus a JSON manifest.
- RF inference helper rejects missing feature names with a useful error.
- Diagnostic wording does not claim correctness when ground truth is missing.

### Canonical Pipeline Foundation Acceptance Criteria

The second implementation slice adds a testable skeleton-to-PipelineRun path before the Streamlit UI and before full MediaPipe upload mode.

Implemented foundation interfaces:

- `src/observatory/pose_quality.py`
  - `evaluate_pose_quality(keypoints, visibilities=None)`
  - returns reliability score, reliability label, warning messages, missing-joint ratios, critical-joint visibility, jitter, max jump, and outlier jump count.
- `src/observatory/pipeline.py`
  - `run_skeleton_pipeline(...)`
  - accepts raw COCO-17 keypoints plus optional visibilities and RF bundle path.
  - stores raw keypoints and normalized keypoints separately.
  - preprocesses to `(64, 17, 2)`.
  - extracts DM features from normalized skeletons.
  - runs RF inference when a bundle is provided.
  - builds safe diagnostics text using optional ground truth.
- `tools/scan_curated_candidates.py`
  - dry-run mode selects a deterministic, class-balanced candidate list from `data/clips`.
  - full mode runs MediaPipe extraction, canonical skeleton pipeline, RF prediction, and saves cached PipelineRun artifacts plus a candidate report.

Verification commands:

```bash
cd BSQAv2
../.venv/Scripts/python.exe -m unittest tests/test_observatory_artifacts.py tests/test_observatory_pipeline.py tests/test_scan_curated_candidates.py
../.venv/Scripts/python.exe -m compileall src/observatory tools/scan_curated_candidates.py tests/test_observatory_artifacts.py tests/test_observatory_pipeline.py tests/test_scan_curated_candidates.py
../.venv/Scripts/python.exe tools/scan_curated_candidates.py --dry-run --max-total 10 --limit-per-class 2 --output ../candidate_scan_smoke.json
```

Acceptance criteria:

- Good synthetic poses score as high reliability.
- Missing critical wrist data lowers reliability and emits a named warning.
- Large joint jumps are reported as pose warnings.
- Skeleton pipeline returns a valid PipelineRun.
- Skeleton pipeline stores raw `(T, 17, 2)` and normalized `(64, 17, 2)` arrays separately.
- Skeleton pipeline runs RF inference from the exported RF bundle.
- Unknown-label diagnostics remain cautious and do not claim correctness.
- Candidate scan dry-run produces a deterministic balanced list without invoking MediaPipe.

### Deep Learning Inference Foundation Acceptance Criteria

The third implementation slice adds the GCN + BiLSTM + Temporal Attention branch to the same artifact/pipeline contract.

Implemented interfaces:

- `src/observatory/dl_inference.py`
  - `load_dl_model(checkpoint_path, device="cpu")`
  - `run_dl_inference(bundle, normalized_keypoints)`
  - supports checkpoints saved directly as state dicts or as dictionaries with `model_state_dict`.
  - strips known training wrapper prefixes such as `inner.` and `module.`.
  - handles older checkpoints trained before joint-attention GCN pooling by falling back to mean pooling when only `gcn.joint_attn_proj.*` is missing.
  - returns prediction probabilities, confidence, quality head output, attention weights, and shape metadata.
- `src/observatory/pipeline.py`
  - `run_skeleton_pipeline(...)` now accepts optional `dl_checkpoint_path` and `dl_device`.
  - stores `attention_weights.npy` when DL inference is enabled.
  - writes `dl_summary`, `dl_shapes`, `dl_quality_score`, and RF/DL branch comparison diagnostics.
- `tools/scan_curated_candidates.py`
  - accepts optional `--dl-checkpoint`.
  - candidate reports can include DL prediction, DL confidence, DL correctness, and RF/DL agreement.

Verification commands:

```bash
cd BSQAv2
../.venv/Scripts/python.exe -m unittest tests/test_observatory_dl_inference.py
../.venv/Scripts/python.exe -m unittest tests/test_observatory_artifacts.py tests/test_observatory_pipeline.py tests/test_scan_curated_candidates.py tests/test_observatory_dl_inference.py
```

Acceptance criteria:

- DL checkpoint loading gives a helpful error when the checkpoint is missing.
- DL inference returns probabilities that sum to approximately 1.
- DL inference returns temporal attention weights shaped `(64, 64)`.
- PipelineRun can include both RF and DL predictions.
- PipelineRun stores attention weights separately from skeleton arrays.
- Branch comparison diagnostics are produced when both RF and DL predictions exist.

### Curated Manifest Builder Acceptance Criteria

The fourth implementation slice turns candidate scan reports into the final curated case bank consumed by Streamlit.

Implemented interface:

- `tools/build_curated_manifest.py`
  - reads a candidate scan report.
  - selects approximately 12 teaching-oriented samples.
  - writes `webapp/artifacts/curated/manifest.json`.
  - preserves required metadata: sample ID, title, ground truth, video path, PipelineRun path, teaching point, diagnosis, tags, RF/DL predictions, pose reliability, and pose warnings.

Selection buckets:

- one representative per stroke class,
- both-model-correct cases,
- RF-correct/DL-wrong disagreement cases,
- pose-warning cases,
- low RF confidence case,
- lift ambiguity case when available,
- high-quality fillers when a bucket cannot be filled.

Verification commands:

```bash
cd BSQAv2
../.venv/Scripts/python.exe -m unittest tests/test_build_curated_manifest.py
../.venv/Scripts/python.exe tools/build_curated_manifest.py --report webapp/artifacts/curated/candidate_scan_20260529_131223.json --target-count 12
```

Acceptance criteria:

- Manifest builder returns the requested target count when enough candidates exist.
- Selection covers all 5 stroke classes.
- Selection includes teaching tags for disagreement, pose warning, and low confidence cases.
- Each manifest entry includes required metadata and manual review status.
- `ArtifactRegistry().list_curated_samples()` loads the generated manifest successfully.

## Major Modules

### Artifact and Registry Module

Responsible for discovering, validating, loading, and saving demo artifacts. It should provide a simple interface for:

- listing curated samples,
- loading a selected sample manifest,
- loading cached PipelineRun artifacts,
- saving new PipelineRun artifacts,
- resolving model, metric, figure, and feature-space assets.

This should be a deep module because many pages need artifact access but should not know storage details.

### Canonical Pipeline Module

Responsible for running the full pipeline from video to predictions:

- load video metadata,
- run MediaPipe pose extraction,
- convert to COCO-17,
- run pose QC,
- preprocess skeleton,
- run DL inference,
- run DM feature extraction,
- run RF inference,
- build diagnosis.

This should be a deep module with a stable interface that returns PipelineRun.

### Pose Quality Module

Responsible for pose diagnostics:

- visibility statistics,
- missing joint ratio,
- critical joint visibility,
- frame-to-frame jitter,
- outlier jumps,
- pose reliability score,
- pose warning messages.

This should be testable independently using synthetic skeleton arrays.

### DL Inference Module

Responsible for:

- loading the GCN + BiLSTM + Attention checkpoint,
- preparing model input,
- running prediction,
- extracting class probabilities,
- extracting attention weights,
- optionally exposing intermediate shapes for Technical View.

### DM Inference Module

Responsible for:

- extracting biomechanical features from normalized skeletons,
- matching saved RF feature names and order,
- handling missing values consistently,
- loading RF artifacts,
- running RF prediction,
- computing current-sample vs class-average comparisons,
- exposing top supporting features.

### Diagnostics Module

Responsible for producing human-readable explanations:

- correctness when ground truth is known,
- cautious fallback when ground truth is unknown,
- RF vs DL agreement/disagreement,
- confidence margin interpretation,
- pose reliability interpretation,
- known class confusion explanation,
- error cause suggestions.

### Graphs Module

Responsible for reusable chart generation:

- pose visibility heatmap,
- missing joint ratio over time,
- jitter graph,
- probability bar charts,
- attention heatmap,
- frame importance curve,
- biomechanical time-series plots,
- mutual information chart,
- feature distribution plots,
- feature correlation heatmap,
- confusion matrix display,
- per-class F1 chart,
- t-SNE or UMAP plots,
- robustness degradation curves.

### UI Components Module

Responsible for reusable Streamlit components:

- shared sidebar,
- explanation cards,
- metric cards,
- label/data quality panel,
- skeleton viewer,
- pipeline step tracker,
- shape tracker,
- probability charts,
- diagnosis panel.

### Sample Preparation Tooling

Responsible for preparing curated samples:

- scan candidate clips from `data/clips`,
- run the canonical pipeline,
- compute pose reliability,
- run DL and RF predictions,
- save candidate reports,
- help select the final 12 curated cases.

This may be implemented as a command-line tool rather than a Streamlit page.

## Page Requirements

### Overview Page

Must show:

- project summary,
- full architecture diagram,
- DL branch summary,
- DM branch summary,
- final model metrics,
- main conclusion about RF vs DL,
- Guided Presentation explanation when enabled.

### Full Pipeline Demo Page

Must show:

- selected video or uploaded video,
- run/load pipeline controls,
- pipeline step tracker,
- stage timing,
- shape tracker,
- pose reliability summary,
- DL prediction,
- RF prediction,
- agreement/disagreement summary,
- label/data quality panel.

### Pose Inspector Page

Must show:

- original video and skeleton overlay,
- frame slider,
- keypoint/visibility table,
- joint visibility heatmap,
- critical joint visibility chart,
- missing joint ratio,
- jitter/outlier jump graph,
- pose reliability score and warnings.

### Deep Learning Inspector Page

Must show:

- architecture explanation,
- tensor shapes in Technical View,
- DL prediction and class probabilities,
- attention heatmap,
- frame importance curve,
- top attention frames,
- interpretation of confidence and attention behavior.

### Data Mining Motion Lab Page

Must show two levels:

1. Global Dataset Mining:
   - class distribution and entropy,
   - mutual information ranking,
   - feature distribution plots,
   - feature correlation heatmap,
   - RF feature importance,
   - t-SNE or UMAP feature space,
   - RF confusion matrix.

2. Local Current-Sample Explanation:
   - biomechanical feature table,
   - feature time-series graphs,
   - RF prediction probabilities,
   - current sample vs class averages,
   - decision tree or rule explanation,
   - local explanation text.

### Error Analysis Lab Page

Must show:

- selected sample correctness if ground truth is known,
- no correctness claim if ground truth is unknown,
- RF vs DL agreement/disagreement,
- probability comparison,
- confidence margin,
- pose reliability contribution,
- class confusion explanation,
- automatic diagnosis report,
- curated error case list.

### Training and Evaluation Page

Must show:

- final DL v5 metrics,
- RF baseline metrics,
- LSTM baseline metrics,
- ablation/run comparison,
- training loss curves where available,
- validation accuracy/F1 curves where available,
- confusion matrices,
- per-class F1 comparison,
- explanation of why no more training is planned.

### Dataset Explorer Page

Must show:

- class distribution,
- source distribution where available,
- curated sample manifest,
- data quality/label trust information,
- sample browser,
- pose quality distribution if computed.

### Robustness Experiment Page

Must show:

- selected clean skeleton,
- controls for artificial noise, wrist dropout, elbow dropout, and frame dropout,
- prediction and confidence under degradation,
- degradation curve,
- explanation that this demonstrates downstream sensitivity to pose quality.

## Testing Decisions

Tests should focus on external behavior and pipeline contracts, not Streamlit rendering details.

Good tests should verify:

- a video or cached artifact can produce a valid PipelineRun,
- PipelineRun contains required fields and valid shapes,
- raw and normalized skeleton arrays are stored separately,
- pose reliability scoring behaves correctly for good, missing, and jittery skeletons,
- preprocessing always returns a 64-frame normalized skeleton,
- DM feature extraction returns feature names matching RF artifacts,
- RF inference fails gracefully when feature names or model artifacts are missing,
- DL inference returns probabilities summing to approximately 1,
- diagnostic text does not claim correctness when ground truth is unknown,
- diagnostic text includes correctness when ground truth is known,
- curated manifest entries have required metadata,
- artifact loading handles missing files with helpful errors,
- graph data preparation functions return expected data shapes.

Modules recommended for tests:

- Canonical Pipeline Module,
- PipelineRun artifact serialization/deserialization,
- Pose Quality Module,
- DM Inference Module,
- Diagnostics Module,
- Sample Registry Module,
- robustness degradation transforms.

Tests do not need to verify exact graph styling, Streamlit layout, or visual appearance. They should verify the data prepared for charts and the decisions produced by diagnosis logic.

## Out of Scope

- Real-time webcam prediction.
- Quality scoring using DTW or biomechanics rules.
- Professional coaching feedback such as technique correction.
- Retraining models inside the Streamlit app.
- Full YouTube download or clip trimming inside the app.
- User accounts, authentication, or saved user history.
- Cloud deployment requirements.
- Mobile app support.
- Replacing MediaPipe with another pose estimation model.
- Claiming MediaPipe pose accuracy without ground-truth keypoint labels.
- Claiming uploaded-video predictions are correct when no ground-truth label is provided.

## Further Notes

The app should present the project honestly and defensibly:

- Random Forest is the strongest Data Mining model and performs better than the final GCN + BiLSTM + Attention model on the current dataset.
- GCN + BiLSTM + Attention remains the main Deep Learning architecture because it demonstrates spatial graph learning, temporal sequence modeling, and attention-based frame importance.
- The project's strongest academic story is not that DL beats all baselines, but that the system compares two approaches on the same pose pipeline: engineered feature mining versus end-to-end spatial-temporal learning.
- Wrong predictions should be treated as analysis opportunities. The app should explain whether the likely cause is pose quality, class ambiguity, low confidence margin, known weak class, feature overlap, or model limitation.
- The lift class deserves special attention because it has the weakest DL F1 score and is a central confusion case.
- The demo should remain visually rich but not cluttered. Graphs should be abundant across the app but limited per page, with additional details in expanders.
- Curated Investigation Mode should be stable enough for live defense. Custom Upload Mode should demonstrate authenticity but must include warnings and graceful failure handling.
- The artifact layer should be prepared before heavy UI implementation, because it will make the app faster, more reliable, and easier to explain.
