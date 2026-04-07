# BSQAv2 — Full Project Plan (v2 — Updated)

> **Goal:** Build the production-grade version of Badminton Stroke Quality Assessment using **hand-rolled Spatial GCN + BiLSTM + Temporal Attention**, with expanded data from YouTube elite players (5 classes), hybrid quality scoring (DTW + biomechanics rules), 5-Fold CV, and a full ablation study comparing architectures.

> **Timeline:** 2 months. Full scope (P0 + P1 + P2) is achievable.

> **Key Design Decision:** We use Spatial GCN for joint relationships and BiLSTM + Attention for temporal modeling. This is NOT ST-GCN — ST-GCN uses temporal convolution, which we deliberately replace with BiLSTM + Attention for better long-range dependencies and explainability. ST-GCN serves as a **literature baseline** for comparison.

---

## Why v2? (Not just more work on v1)

| Aspect | BSQAv1 (Keep as-is) | BSQAv2 (New folder) |
|--------|---------------------|---------------------|
| **Purpose** | Proof of concept, prototype | Production-grade, report-ready |
| **Model** | LSTM baseline only | GCN + BiLSTM + Attention |
| **Data** | 150 samples, 3 classes (Kaggle only) | 600+ samples, 5 classes (Kaggle + YouTube elite) |
| **Output** | Classification only | Classification + Quality Score + Feedback |
| **Report Value** | "Our baseline" reference | "Our proposed method" — the main contribution |

> [!IMPORTANT]
> Keep BSQAv1 untouched. Your academic report needs v1 results as the **baseline comparison**. v2 proves your improvements.

---

## User Review Required

> [!NOTE]
> **Resolved Decisions (Apr 2):**
> - ✅ **5 classes:** smash, clear, drop_shot, net_shot, lift
> - ✅ **Hand-rolled GCN** (no torch-geometric) — simpler, explainable, no dependency issues
> - ✅ **5-Fold cross-validation** for robust evaluation
> - ✅ **2-month timeline** — full scope (all phases) is achievable
> - ✅ **Augmentation order fixed:** Split first → augment train only → val/test untouched
> - ✅ **Hybrid quality scoring** (DTW + rules, no supervised quality labels needed)

---

## Proposed Changes

### Phase 0: Project Setup & Shared Data

#### [NEW] `BSQAv2/` — New project root

```
BSQA/
├── BSQAv1/                    # Keep untouched (baseline reference)
├── BSQAv2/                    # New project
│   ├── src/
│   │   ├── config.py          # Updated config (5 classes, GCN params)
│   │   ├── data/
│   │   │   ├── dataset.py     # Updated loader (v1 + v2 CSVs, K-Fold splits)
│   │   │   ├── preprocessing.py   # Reuse from v1
│   │   │   ├── skeleton.py    # Reuse from v1 + adjacency matrix builder for GCN
│   │   │   └── augmentation.py    # [NEW] Time warp, mirror, noise (TRAIN ONLY)
│   │   ├── models/
│   │   │   ├── gcn.py         # [NEW] Hand-rolled GCN (A × H × W, no PyG)
│   │   │   ├── bilstm.py      # [NEW] BiLSTM temporal module
│   │   │   ├── attention.py   # [NEW] Temporal Attention module
│   │   │   ├── gcn_bilstm.py  # [NEW] GCN + BiLSTM (ablation config A)
│   │   │   ├── gcn_bilstm_attn.py  # [NEW] Full model (GCN + BiLSTM + Attn)
│   │   │   └── lstm_baseline.py    # Copy from v1 for ablation baseline
│   │   ├── quality/
│   │   │   ├── dtw_scorer.py  # [NEW] DTW similarity scoring
│   │   │   ├── rules.py       # [NEW] Biomechanics rule-based scoring
│   │   │   └── hybrid.py      # [NEW] Combined DTW + rules scorer
│   │   └── utils/
│   │       ├── visualization.py   # Enhanced from v1
│   │       ├── video_to_csv.py    # [NEW] MediaPipe extraction pipeline
│   │       └── metrics.py     # [NEW] Classification + AQA metrics
│   ├── tools/
│   │   ├── download_clips.py  # [NEW] yt-dlp batch downloader
│   │   └── trim_clips.py      # [NEW] ffmpeg batch trimmer
│   ├── train.py               # Updated training script (multi-model)
│   ├── predict.py             # Updated prediction (classification + quality)
│   ├── evaluate.py            # [NEW] Full evaluation + report generation
│   ├── notebooks/
│   │   ├── 01_eda_extended.ipynb      # EDA on expanded dataset
│   │   ├── 02_ablation_study.ipynb    # Model comparison notebook
│   │   └── 03_quality_analysis.ipynb  # Quality scoring analysis
│   ├── data/
│   │   ├── kaggle/            # Symlink or copy from BSQAv1 dataset
│   │   ├── youtube/           # New YouTube-extracted CSVs
│   │   └── metadata.csv       # Source tracking for all samples
│   ├── checkpoints/
│   ├── runs/                  # TensorBoard logs
│   ├── results/               # Evaluation outputs, figures, tables
│   └── requirements.txt
└── .agent/                    # Keep shared agent config
```

---

### Phase 1: Data Expansion (Days 1-3)

> **Goal:** Expand from 150 samples / 3 classes → 600+ samples / 5 classes

#### [NEW] `BSQAv2/tools/download_clips.py`
- Reads a timestamp log CSV (manually created while watching YouTube)
- Uses `yt-dlp` to batch-download videos
- Uses `ffmpeg` to trim clips based on start/end timestamps
- Output: `clips/` folder with named stroke clips

#### [NEW] `BSQAv2/src/utils/video_to_csv.py`
- Takes a folder of video clips as input
- Runs MediaPipe Pose on each clip
- Maps MediaPipe 33 keypoints → COCO-17 format
- Applies quality control filters:
  - Confidence threshold (visibility > 0.5 for critical joints)
  - Jump detection (> 100px between frames = flag)
  - Missing joint threshold (> 30% missing = discard)
- Outputs CSV files matching your v1 schema (37 columns: id, type_of_shot, frame_count, kpt_0_x, kpt_0_y, ...)

#### Data Targets

| Stroke | Existing (v1) | New (YouTube) | Total Target |
|--------|---------------|---------------|--------------|
| Smash | 50 | 100-150 | 150-200 |
| Lift | 50 | 100-150 | 150-200 |
| Net shot | 50 | 100-150 | 150-200 |
| Clear | 0 | 100-150 | 100-150 |
| Drop shot | 0 | 100-150 | 100-150 |
| **Total** | **150** | **500-750** | **650-900** |

#### [NEW] `BSQAv2/src/data/augmentation.py`

> [!IMPORTANT]
> **Augmentation Order (CRITICAL):**
> 1. Load all original data
> 2. Split into Train / Val / Test **FIRST** (within each K-Fold)
> 3. Apply augmentation to **TRAIN set ONLY**
> 4. Val and Test remain **original, untouched data**
> 
> This prevents data leakage — augmented copies of a training sample must never appear in validation or test.

**Augmentation techniques (applied to train only):**
- **Time warping:** Speed up/slow down by 0.8x-1.2x (×3-5 per sample)
- **Mirror flip:** Horizontal flip for left/right hand variation (×2)
- **Joint noise:** Small Gaussian noise on coordinates (×2-3)
- **Frame dropout:** Randomly remove 5-10% of frames (×2)
- Effective training set after augmentation: ~3,000-6,000+ samples
- Val/Test remain at original ~130 / ~130 samples (across 5-Fold)

---

### Phase 2: Core Model Architecture (Days 4-8)

> **Goal:** Implement GCN + BiLSTM + Attention in modular, ablation-friendly structure

#### [NEW] `BSQAv2/src/models/gcn.py` — Hand-Rolled Spatial GCN

The GCN takes skeleton graph data and learns spatial relationships between joints.
**No torch-geometric dependency** — implemented from scratch using adjacency matrix math.

> [!NOTE]
> **Why NOT ST-GCN?** ST-GCN combines spatial GCN with *temporal convolution* (1D conv over frames). We deliberately replace the temporal convolution with BiLSTM + Attention, which gives us:
> - Better long-range temporal dependencies (BiLSTM)
> - Explainable frame importance (Attention weights)
> - A genuine architectural novelty for our report
>
> ST-GCN is included in the ablation study as a **literature baseline**.

**GCN Layer Math (hand-rolled):**
```
H_out = σ(D̃⁻¹/² × Ã × D̃⁻¹/² × H_in × W)

Where:
- Ã = A + I  (adjacency + self-loops, 17×17, FIXED)
- D̃ = degree matrix of Ã
- H_in = joint features (17 × input_dim)
- W = learnable weight matrix
- σ = ReLU activation
```

**Architecture per frame:**
```
Input: (17 joints, 2 coords)
  ↓
GCN Layer 1: (17, 2) → (17, 64)     # Each joint aggregates neighbor info
  ↓ BatchNorm + ReLU + Dropout
GCN Layer 2: (17, 64) → (17, 128)    # Deeper spatial understanding
  ↓ BatchNorm + ReLU + Dropout
GCN Layer 3: (17, 128) → (17, 128)   # Refine
  ↓
Mean Pool over joints: (17, 128) → (128)   # Aggregate all joints
  ↓
Output per frame: (128,)              # One spatial feature vector per frame
```

**Key component:** The 17×17 adjacency matrix built from `skeleton.py` SKELETON_EDGES

#### [NEW] `BSQAv2/src/models/bilstm.py` — Bidirectional LSTM

Takes the sequence of GCN-enhanced frames and models temporal dynamics.

**Architecture:**
```
Input: (64 frames, 128)              # After GCN mean-pool over joints
  ↓
BiLSTM Layer 1: hidden=128 → output=256     # Forward(128) + Backward(128)
  ↓ Dropout
BiLSTM Layer 2: hidden=128 → output=256
  ↓
Output: (64, 256)                            # Temporally-enriched features
```

#### [NEW] `BSQAv2/src/models/attention.py` — Temporal Attention

Learns which frames are most important (e.g., impact frame should get high weight).

**Architecture:**
```
Input: (64, 256)                             # BiLSTM output
  ↓
Multi-Head Self-Attention (4 heads)
  Q = Linear(256 → 256)
  K = Linear(256 → 256)
  V = Linear(256 → 256)
  ↓
Attention weights: (64, 64)                  # Frame-to-frame importance
  ↓
Weighted output → Global pool
  ↓
Output: (256,)                               # Single context vector
```

**Bonus for report:** Attention weights are visualizable — you can show which frames the model focused on (should highlight impact/swing frames).

#### [NEW] `BSQAv2/src/models/gcn_bilstm_attn.py` — Full Model

The complete architecture combining all modules:

```
┌─────────────────────────────────────────────────────────────────┐
│                    FULL MODEL ARCHITECTURE                       │
│                                                                  │
│  Input: (batch, 64 frames, 17 joints, 2 coords)                │
│    │                                                             │
│    ▼                                                             │
│  ┌─────────────┐                                                │
│  │   GCN ×3    │  Per-frame spatial feature extraction          │
│  │  (17,2)→    │                                                │
│  │  (17,128)   │                                                │
│  └─────┬───────┘                                                │
│        │ Mean Pool joints: (17, 128) → (128) per frame          │
│        ▼                                                         │
│  ┌─────────────┐                                                │
│  │  BiLSTM ×2  │  Temporal sequence modeling                    │
│  │ (64,128)→   │                                                │
│  │ (64,256)    │                                                │
│  └─────┬───────┘                                                │
│        ▼                                                         │
│  ┌─────────────┐                                                │
│  │  Temporal   │  Focus on important frames                     │
│  │  Attention  │                                                │
│  │ (64,256)→   │                                                │
│  │ (256)       │                                                │
│  └─────┬───────┘                                                │
│        │                                                         │
│   ┌────┴────┐                                                   │
│   ▼         ▼                                                    │
│ ┌───────┐ ┌────────┐                                            │
│ │Class  │ │Quality │                                            │
│ │Head   │ │Head    │                                            │
│ │→(5,)  │ │→(1,)   │                                            │
│ └───────┘ └────────┘                                            │
└─────────────────────────────────────────────────────────────────┘
```

#### Ablation-Friendly Design

Each module is a separate file so you can mix and match:

| Configuration | Files Used | Purpose |
|---|---|---|
| **LSTM Baseline** | `lstm_baseline.py` | Baseline from v1 (already have results) |
| **BiLSTM Baseline** | `bilstm.py` only | Show bidirectional improvement |
| **GCN + LSTM** | `gcn.py` + standard LSTM | Show GCN spatial improvement |
| **GCN + BiLSTM** | `gcn.py` + `bilstm.py` | Show BiLSTM temporal improvement |
| **GCN + BiLSTM + Attention** | All three | **Full proposed model** |
| **ST-GCN (cited)** | Not implemented — cite paper results | Literature reference only |

---

### Phase 3: Training Pipeline (Days 9-11)

> **Goal:** Train all ablation configurations, log everything to TensorBoard

#### [MODIFY] `BSQAv2/train.py`
- Support multiple model architectures via `--model` flag
- Mixed precision training (for GTX 1650 4GB VRAM)
- Data augmentation integration
- **5-Fold stratified cross-validation** (critical with ~650 samples)
- Augmentation applied **inside each fold, to train split only**
- Save best model per configuration

```bash
# Usage examples:
python train.py --model lstm_baseline --epochs 100
python train.py --model gcn_bilstm --epochs 100
python train.py --model gcn_bilstm_attn --epochs 100 --augment
```

#### Training Hyperparameters (v2)

| Parameter | v1 Value | v2 Value | Reason |
|-----------|----------|----------|--------|
| Sequence Length | 64 | 64 | Keep consistent for comparison |
| GCN Hidden Dim | — | 128 | Balance speed/accuracy |
| GCN Layers | — | 3 | More may oversmooth |
| GCN Pool Strategy | — | Mean Pool | Average over 17 joints → single vector per frame |
| BiLSTM Hidden | 128 | 128 | (output=256 due to bidirectional) |
| Attention Heads | — | 4 | Standard for small models |
| Learning Rate | 1e-3 | 1e-3 (with cosine decay) | Better convergence |
| Batch Size | 8 | 16 (if data is larger) | More data allows bigger batches |
| Epochs | 100 | 100 | With early stopping patience=15 |
| Dropout | 0.3 | 0.3 | Consistent |
| Num Classes | 3 | 5 | Adding "clear" + "drop_shot" |

---

### Phase 4: Quality Assessment — Hybrid Scoring (Days 12-14)

> **Goal:** Implement DTW + Rule-based quality scoring that runs AFTER classification

#### [NEW] `BSQAv2/src/quality/dtw_scorer.py`
- Takes user's skeleton sequence + matched pro references (same stroke type)
- Computes DTW distance between user and each pro reference
- Returns similarity score 0-100 (higher = closer to pro)
- Supports FastDTW for efficiency if needed

#### [NEW] `BSQAv2/src/quality/rules.py`
- Biomechanics rule engine per stroke type
- **Smash rules:** elbow angle at impact (120-170°), wrist velocity, contact height above shoulder
- **Clear rules:** full arm extension, contact height, body rotation
- **Drop shot rules:** deceptive preparation (similar to smash), gentle racket deceleration, steep downward trajectory
- **Net shot rules:** minimal arm swing, soft wrist, racket face angle
- **Lift rules:** upward trajectory, wrist supination, low contact point
- Each rule returns: score (0-100), specific textual feedback
- Sources cited: Phomsoupha & Laffaye (2015), BWF coaching guidelines

#### [NEW] `BSQAv2/src/quality/hybrid.py`
- Combines DTW score (40% weight) + Rule-based score (60% weight)
- Returns:
  ```python
  {
      "stroke_type": "smash",
      "classification_confidence": 0.94,
      "quality_score": 78,
      "dtw_score": 72,
      "rule_scores": {
          "elbow_angle": 85,
          "wrist_velocity": 65,
          "contact_height": 90
      },
      "feedback": [
          "Elbow angle 115° (ideal: 120-170°) — raise elbow higher during swing",
          "Wrist velocity below pro average — snap wrist more sharply at impact"
      ],
      "attention_weights": [...],  # For visualization
  }
  ```

---

### Phase 5: Evaluation & Ablation Study (Days 15-17)

> **Goal:** Generate all results needed for your academic report

#### [NEW] `BSQAv2/evaluate.py`
- Runs all models on the test set
- Generates:
  - Confusion matrices per model
  - Accuracy / F1-Score comparison table
  - Training curves comparison plot
  - Attention weight visualizations
  - Quality score distribution analysis

#### Ablation Study Results Table (What Your Report Needs)

| Model | Accuracy (5-Fold avg) | F1-Score | Params | Notes |
|-------|----------|----------|--------|---------------|-------|
| LSTM Baseline | (from v1) | (from v1) | ~200K | Baseline from BSQAv1 |
| BiLSTM Baseline | ? | ? | ~400K | Bidirectional upgrade |
| GCN + LSTM | ? | ? | ~350K | +GCN spatial |
| GCN + BiLSTM | ? | ? | ~500K | +Bidirectional temporal |
| **GCN + BiLSTM + Attn** | ? | ? | ~550K | **Full proposed model** |
| ST-GCN (cited) | — (cite paper) | — | ~300K | Literature reference only |

#### Quality Assessment Validation
- Select 20 samples: 10 "known good" (pro data) + 10 "artificially degraded" (add noise/offset to pro data to simulate bad technique)
- Show that the system correctly scores pro data higher than degraded data
- Visualize attention weights on good vs bad strokes

---

### Phase 6: Web Demo (Days 18-20, IF TIME PERMITS)

> **Goal:** Streamlit app for live demonstration

#### [NEW] `BSQAv2/webapp/`
- Upload video → MediaPipe extraction → Model prediction → Quality score + feedback
- Show attention heatmap on video timeline
- Show skeleton overlay on video frames
- Display rule-based feedback with specific suggestions

---

## Open Questions — All Resolved ✅

| # | Question | Decision (Apr 2) |
|---|----------|-------------------|
| 1 | How many stroke types? | **5 classes:** smash, clear, drop_shot, net_shot, lift |
| 2 | GCN implementation? | **Hand-rolled** (adjacency matrix × features × weights, no PyG) |
| 3 | Cross-validation? | **5-Fold stratified** CV |
| 4 | Timeline? | **2 months** — full scope achievable |
| 5 | Augmentation order? | **Split first → augment train only** (user-corrected) |
| 6 | ST-GCN? | **Literature baseline only** — our model uses BiLSTM+Attn instead |

---

## Verification Plan

### Automated Tests
```bash
# Phase 1: Data pipeline
python src/utils/video_to_csv.py --input clips/ --verify  # Visual spot-check

# Phase 2: Model architecture
python -c "from src.models.gcn_bilstm_attn import FullModel; ..."  # Shape test

# Phase 3: Training (per fold)
python train.py --model gcn_bilstm_attn --epochs 5 --quick-test --fold 0

# Phase 4: Quality scoring
python -c "from src.quality.hybrid import HybridScorer; ..."  # Unit test

# Phase 5: Full evaluation
python evaluate.py --output results/ --all-models --kfold 5
```

### Manual Verification
- TensorBoard: training curves converge per fold, no anomalies
- Confusion matrix: no class completely misclassified across folds
- Attention weights: impact frames get highest attention
- Quality scores: pro samples > degraded samples consistently
- Augmentation sanity: verify val/test sets contain NO augmented samples

---

## What Carries Over from v1 (Don't Rewrite)

| Component | Reuse Strategy |
|-----------|---------------|
| `preprocessing.py` | Copy + add augmentation support |
| `skeleton.py` | Copy + add adjacency matrix builder for GCN |
| `visualization.py` | Copy + extend with attention visualization |
| Original dataset | Symlink or copy the 3 CSV files |
| LSTM results | Reference in ablation study (don't retrain) |

---

## What's New in v2 (Must Build)

| Week | Phase | Components | Effort |
|------|-------|-----------|--------|
| 1-2 | **Data Collection** | YouTube prospecting, download, trim, MediaPipe extract, QC | 5-7 days |
| 2 | **Data Pipeline** | augmentation.py (train-only), dataset.py (K-Fold), EDA notebook | 2 days |
| 3 | **Model Architecture** | Hand-rolled GCN, BiLSTM, Temporal Attention, full model assembly | 3-4 days |
| 4 | **Training Pipeline** | Multi-model train.py, 5-Fold CV loop, mixed precision, TensorBoard | 2-3 days |
| 4-5 | **Ablation Training** | Train all 5 configurations × 5 folds (GPU time, mostly automated) | 3-5 days |
| 5-6 | **Quality Assessment** | DTW scorer, biomechanics rules, hybrid combiner | 3-4 days |
| 6-7 | **Evaluation** | evaluate.py, ablation tables, confusion matrices, attention viz | 2-3 days |
| 7-8 | **Web Demo** | Streamlit app (upload video → prediction + quality + feedback) | 3-4 days |
| 8 | **Report & Polish** | Final figures, result tables, clean documentation | 2-3 days |

**Total estimated: ~6-8 weeks of active work, comfortably within 2-month timeline**
