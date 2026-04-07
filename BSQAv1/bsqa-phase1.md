# BSQA Phase 1: Proof of Concept

> **Goal:** Build a working 3-class stroke classifier (Smash, Lift, Net shot) with ≥85% accuracy using the Kaggle badminton dataset.

---

## Overview

This phase establishes the core ML pipeline: data loading, preprocessing, baseline model (LSTM), and training. No web demo yet—focus is on validating the dataset and achieving target accuracy.

| Property | Value |
|----------|-------|
| **Timeline** | Week 1 |
| **Project Type** | ML/Python |
| **Primary Agent** | `backend-specialist` + custom ML focus |
| **Dataset** | Kaggle 17-keypoint skeleton data (~10K samples) |
| **Classes** | Smash, Lift, Net shot |
| **Target Accuracy** | ≥85% |

---

## Success Criteria

| # | Criterion | How to Verify |
|---|-----------|---------------|
| 1 | Data loads without errors | Run `python -c "from src.data import load_data; load_data()"` |
| 2 | Preprocessing normalizes coordinates | Visual inspection of normalized coordinates (should be ~[-1, 1]) |
| 3 | Model trains without errors | Training loop completes, loss decreases |
| 4 | Validation accuracy ≥85% | `python train.py` prints final accuracy |
| 5 | Model can make predictions on new data | `python predict.py --input sample.csv` outputs class label |

---

## Tech Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **Language** | Python 3.9+ | ML ecosystem |
| **ML Framework** | PyTorch 2.0+ | GNN support via PyG |
| **Data** | Pandas, NumPy | CSV handling, array ops |
| **Visualization** | Matplotlib | Training curves, skeleton viz |
| **Experiment Tracking** | TensorBoard | Loss/accuracy logging |

---

## File Structure

```
BSQAv1/
├── src/
│   ├── __init__.py
│   ├── config.py              # Hyperparameters, skeleton config
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py         # PyTorch Dataset class
│   │   ├── preprocessing.py   # Normalization, missing keypoint handling
│   │   └── skeleton.py        # 17-keypoint definitions, adjacency matrix
│   ├── models/
│   │   ├── __init__.py
│   │   └── lstm_baseline.py   # Baseline LSTM model
│   └── utils/
│       ├── __init__.py
│       └── visualization.py   # Skeleton plotting, training curves
├── train.py                   # Main training script
├── predict.py                 # Inference script
├── notebooks/
│   └── 01_eda.ipynb           # Exploratory data analysis
├── runs/                      # TensorBoard logs (gitignored)
├── checkpoints/               # Model checkpoints (gitignored)
└── requirements.txt
```

---

## Task Breakdown

### Phase 1.1: Setup & Data Loading

| # | Task | Verify |
|---|------|--------|
| 1.1.1 | Create project structure (`src/`, `notebooks/`, etc.) | `ls -la src/` shows expected dirs |
| 1.1.2 | Create `requirements.txt` with dependencies | `pip install -r requirements.txt` succeeds |
| 1.1.3 | Create `src/config.py` with hyperparameters | Import without errors |
| 1.1.4 | Create `src/data/skeleton.py` with 17-keypoint definitions | Run `python -c "from src.data.skeleton import SKELETON_EDGES; print(len(SKELETON_EDGES))"` → 16 edges |

---

### Phase 1.2: Preprocessing Pipeline

| # | Task | Verify |
|---|------|--------|
| 1.2.1 | Create `src/data/preprocessing.py` with normalization (hip-centered, torso-scaled) | Unit test: input pixel coords → output normalized coords in [-1, 1] |
| 1.2.2 | Handle missing keypoints (fill with previous frame value) | Unit test: input with 0.0 values → output with interpolated values |
| 1.2.3 | Create `src/data/dataset.py` (PyTorch Dataset, loads CSVs, applies preprocessing) | `len(dataset)` returns expected sample count |

---

### Phase 1.3: Baseline Model

| # | Task | Verify |
|---|------|--------|
| 1.3.1 | Create `src/models/lstm_baseline.py` (2-layer LSTM + FC head) | Forward pass: `(batch, seq, features)` → `(batch, 3)` logits |
| 1.3.2 | Create `train.py` with training loop, validation, TensorBoard logging | Script runs, TensorBoard shows loss curve |

---

### Phase 1.4: Training & Evaluation

| # | Task | Verify |
|---|------|--------|
| 1.4.1 | Train model on Kaggle dataset (80/20 split) | `python train.py` completes without errors |
| 1.4.2 | Evaluate validation accuracy | Accuracy ≥85% printed to console |
| 1.4.3 | Create `predict.py` for single-sample inference | `python predict.py --input test_sample.csv` outputs class label |
| 1.4.4 | Create EDA notebook (`notebooks/01_eda.ipynb`) with dataset analysis | Notebook runs, shows class distribution, sample skeletons |

---

## Dependencies Between Tasks

```mermaid
graph LR
    A[1.1 Setup] --> B[1.2 Preprocessing]
    B --> C[1.3 Model]
    C --> D[1.4 Training]
```

All tasks in the same phase can be done in parallel. Cross-phase dependencies are sequential.

---

## Phase X: Verification

### Automated Tests

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run unit tests (if created)
pytest tests/ -v

# 3. Quick training test (1 epoch)
python train.py --epochs 1 --quick-test

# 4. Full training
python train.py --epochs 100
```

### Manual Verification

| Check | How |
|-------|-----|
| Data loads correctly | Open EDA notebook, verify sample counts |
| Skeleton visualization | Plot a sample skeleton, verify joint positions make sense |
| Training convergence | Check TensorBoard, loss should decrease |
| Final accuracy | Console output shows ≥85% val accuracy |

---

## Risk Mitigation

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Dataset too noisy | Medium | Start with data cleaning, filter high-missing samples |
| LSTM underperforms | Medium | This is baseline; Phase 2 adds GCN for improvement |
| Hardware limits | Low | Use small batch size (8), no large models yet |

---

## Notes

- This plan focuses on **classification only**. Quality scoring is Phase 3.
- Web demo is Phase 4 (Streamlit).
- Phase 2 will add GCN layers to improve spatial modeling.

---

## Revision History

| Date | Change |
|------|--------|
| 2026-01-28 | Initial plan created |
