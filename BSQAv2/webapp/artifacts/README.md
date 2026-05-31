# Pipeline Observatory Artifacts

This directory stores cached assets for the BSQAv2 Streamlit Pipeline Observatory.

Generated artifacts are intentionally separated from source code:

- `curated/` - curated sample manifest and final case bank metadata
- `pipeline_runs/` - cached `PipelineRun` directories, each containing `run.json` and `.npy` arrays
- `models/` - loadable inference bundles, including the Random Forest joblib bundle
- `metrics/` - copied or normalized metrics JSON/CSV used by dashboards
- `figures/` - reusable static figures from training, RF baseline, and DM analysis
- `feature_space/` - precomputed t-SNE/UMAP coordinates and feature-space metadata

The artifact contract is implemented in `src/observatory/schema.py` and loaded via
`src/observatory/artifacts.py`.

Recommended first artifact export:

```bash
cd BSQAv2
../.venv/Scripts/python.exe src/data/rf_baseline.py --export-artifact
```

This writes `models/rf_baseline/rf_model_bundle.joblib` with the fitted RF model,
label encoder, exact feature order, median imputation values, and class averages.
