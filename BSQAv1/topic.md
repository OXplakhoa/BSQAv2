# Badminton Stroke Quality Assessment using Deep Learning

> **Purpose**: This document serves as the comprehensive anchor for the project. It contains all theoretical foundations, implementation details, and learnings to be referenced throughout development and for future iterations.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement & Innovation](#2-problem-statement--innovation)
3. [Theoretical Foundation](#3-theoretical-foundation)
4. [System Architecture](#4-system-architecture)
5. [Datasets](#5-datasets)
6. [Quality Assessment Methodology](#6-quality-assessment-methodology)
7. [Evaluation Strategy](#7-evaluation-strategy)
8. [Development Workflow](#8-development-workflow)
9. [Prototype Phases](#9-prototype-phases)
10. [Technical Specifications](#10-technical-specifications)
11. [Risk Mitigation & Fallback Plans](#11-risk-mitigation--fallback-plans)
12. [References & Resources](#12-references--resources)
13. [Lessons Learned Log](#13-lessons-learned-log)

---

## 1. Project Overview

| Field | Details |
|-------|---------|
| **Title (EN)** | Badminton Stroke Quality Assessment using Graph Neural Networks with Attention Mechanism |
| **Title (VN)** | Đánh giá chất lượng cú đánh cầu lông dựa trên mạng nơ-ron đồ thị với cơ chế attention |
| **Domain** | Deep Learning, Computer Vision, Sports Analytics |
| **Timeline** | 4 weeks |
| **Team Size** | 3 (primarily individual work) |
| **Hardware** | GTX 1650 (4GB VRAM), i5-12400F, 32GB RAM |

### Project Goals

| Goal | Priority | Metric |
|------|----------|--------|
| **Primary**: Stroke Classification | P0 | Accuracy ≥ 85% |
| **Secondary**: Quality Scoring | P1 | Correlation ≥ 0.7 with pro reference |
| **Tertiary**: Web Demo | P2 | Functional prototype |

---

## 2. Problem Statement & Innovation

### 2.1 Problem

In badminton, stroke quality (smash, clear, drop, net shot) depends on body posture, joint angles, and timing. Currently, quality assessment relies on **subjective coach observation**, which is:
- Expensive (coach fees)
- Not scalable (1 coach : few students)
- Inconsistent (different coaches, different opinions)

### 2.2 Proposed Solution

An AI system that:
1. Extracts skeleton keypoints from video
2. Analyzes spatial relationships between joints (using GCN)
3. Identifies important temporal frames (using Attention)
4. Outputs quality score + actionable feedback

### 2.3 Innovation Points

| Aspect | Traditional Approach | Our Approach |
|--------|---------------------|--------------|
| **Input Representation** | Raw video / CNN features | Skeleton Graph |
| **Spatial Modeling** | CNN (grid-based) | GCN (graph-based) |
| **Temporal Modeling** | LSTM (sequential) | Temporal Attention (parallel) |
| **Output** | Action Classification only | Classification + Quality Score |
| **Explainability** | Black box | Attention visualization + specific feedback |

---

## 3. Theoretical Foundation

### 3.1 Graph Convolutional Networks (GCN)

**Core Idea**: Treat skeleton as a graph where joints are nodes and bones are edges.

```
Human Skeleton Graph:
        [Head]
           │
       [Neck]
      /      \
[L_Shoulder] [R_Shoulder]
     │              │
[L_Elbow]      [R_Elbow]
     │              │
[L_Wrist]      [R_Wrist]  ← Critical for badminton strokes
```

**GCN Formula**:
```
H^(l+1) = σ(D^(-1/2) × A × D^(-1/2) × H^(l) × W^(l))

Where:
- A = Adjacency matrix (which joints connect to which)
- D = Degree matrix
- H = Node features (joint positions)
- W = Learnable weights
- σ = Activation function (ReLU)
```

**Why GCN for Skeleton?**
- Skeleton has natural graph structure (not grid like images)
- GCN learns that shoulder affects elbow, elbow affects wrist
- More parameter-efficient than CNN on skeleton data

### 3.2 Temporal Attention Mechanism

**Core Idea**: Not all frames are equally important. Focus on critical moments.

```
Badminton Smash Timeline:
──────────────────────────────────────────────────────────►
[Prepare][Wind-up][SWING][IMPACT][Follow-through]
           Low      HIGH   VERY HIGH    Medium
        importance                   importance

Attention learns to focus on SWING and IMPACT frames.
```

**Self-Attention Formula**:
```
Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V

Where:
- Q = Query (what am I looking for?)
- K = Key (what do I have?)
- V = Value (what is the content?)
- d_k = dimension for scaling
```

**Why Attention over LSTM?**
| Aspect | LSTM | Temporal Attention |
|--------|------|-------------------|
| Processing | Sequential | Parallel |
| Long-range dependencies | Struggles | Handles well |
| Training speed | Slow | Fast |
| Explainability | Low | High (attention weights) |
| Memory for 90 frames | OK | OK (90² = 8,100) |

### 3.3 ST-GCN (Baseline Reference)

**Spatial-Temporal Graph Convolutional Network**:
- Combines Spatial GCN (inter-joint) + Temporal Convolution (inter-frame)
- Published: AAAI 2018, 5000+ citations
- Our improvement: Replace Temporal Convolution with Temporal Attention

---

## 4. System Architecture

### 4.1 High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INPUT                                   │
│                    (Video: 1-3 seconds)                             │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1: POSE ESTIMATION                                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  MediaPipe Pose                                              │   │
│  │  • Input: Video frames                                       │   │
│  │  • Output: 33 keypoints × T frames × (x, y, z, visibility)  │   │
│  │  • Runs on CPU (no GPU needed)                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 2: PREPROCESSING                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  • Normalize coordinates (center at hip, scale by torso)    │   │
│  │  • Handle missing keypoints (interpolation)                 │   │
│  │  • Pad/truncate to fixed length (e.g., 64 frames)          │   │
│  │  • Build adjacency matrix                                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 3: SPATIAL GRAPH CONVOLUTION                                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Graph Convolutional Network                                 │   │
│  │  • Learn spatial relationships between joints               │   │
│  │  • Multiple GCN layers (e.g., 3 layers)                     │   │
│  │  • Output: Enhanced joint features per frame                │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 4: TEMPORAL ATTENTION                                        │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Multi-Head Self-Attention                                   │   │
│  │  • Learn which frames are important                         │   │
│  │  • Attention weights visualizable                           │   │
│  │  • Output: Temporally-weighted features                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 5: OUTPUT HEADS                                              │
│  ┌──────────────────────────┐  ┌────────────────────────────────┐  │
│  │  Classification Head     │  │  Quality Scoring Head          │  │
│  │  • FC layers             │  │  • Similarity to pro reference │  │
│  │  • Softmax               │  │  • OR: Rule-based metrics      │  │
│  │  • Output: Stroke type   │  │  • Output: Score 0-100         │  │
│  └──────────────────────────┘  └────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼ I j
┌─────────────────────────────────────────────────────────────────────┐
│  OUTPUT                                                              │
│  • Stroke Type: Smash / Clear / Drop / Net / Lift                   │
│  • Quality Score: 78/100                                            │
│  • Feedback: "Elbow angle 105° (ideal: 120°) - raise elbow higher" │
│  • Attention Visualization: Highlighted important frames            │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Component Specifications

| Component | Technology | Input Shape | Output Shape |
|-----------|------------|-------------|--------------|
| Pose Estimation | MediaPipe | (H, W, 3) per frame | (33, 4) per frame |
| Preprocessing | NumPy | (T, 33, 4) | (64, 33, 3) |
| Spatial GCN | PyTorch Geometric | (64, 33, 3) | (64, 33, 128) |
| Temporal Attention | PyTorch | (64, 33, 128) | (33, 128) |
| Classification Head | PyTorch | (33 × 128) | (5,) |
| Quality Head | PyTorch | (33 × 128) | (1,) |

---

## 5. Datasets

### 5.1 Primary Dataset: Kaggle Badminton Motion Data

| Property | Value |
|----------|-------|
| **Name** | Motion Data for Badminton Shots |
| **Source** | Kaggle |
| **URL** | https://www.kaggle.com/datasets/motion-data-badminton-shots |
| **Content** | Skeleton coordinates (joints: head, shoulders, elbows, wrists, hips, knees, feet) |
| **Stroke Types** | Smash, Lift, Net shot |
| **Format** | CSV |
| **Pre-extracted** | Yes (no pose estimation needed) |

**Pros**: Ready to use, professional players
**Cons**: May have noise, no quality labels

### 5.2 Secondary Dataset: ShuttleSet

| Property | Value |
|----------|-------|
| **Name** | ShuttleSet / ShuttleSet22 |
| **Source** | CoachAI Project (NYCU Taiwan) |
| **URL** | https://github.com/wywyWang/CoachAI-Projects |
| **Content** | 44 high-level singles matches (2018-2021) |
| **Annotations** | Stroke type, hitting location, player position |
| **Format** | Video + CSV metadata |

**Pros**: Largest public badminton dataset, detailed annotations
**Cons**: Need to extract skeleton from video

### 5.3 Pre-training Dataset: NTU RGB+D

| Property | Value |
|----------|-------|
| **Name** | NTU RGB+D 60 |
| **URL** | https://rose1.ntu.edu.sg/dataset/actionRecognition/ |
| **Content** | 56,880 clips, 60 action classes |
| **Skeleton** | 25 keypoints × 3D coordinates |
| **Size** | ~5.8 GB (skeleton only) |

**Use Case**: Pre-train GCN on general actions, then fine-tune on badminton.

---

## 6. Quality Assessment Methodology

### 6.1 Approach 1: Similarity-Based (Recommended)

**Concept**: Compare user stroke to professional player reference.

```python
def similarity_scoring(user_embedding, pro_embeddings):
    """
    user_embedding: (256,) - encoded user stroke
    pro_embeddings: (N, 256) - encoded pro strokes of same type
    """
    # Cosine similarity with all pro references
    similarities = F.cosine_similarity(
        user_embedding.unsqueeze(0), 
        pro_embeddings
    )
    # Score = average similarity, scaled to 0-100
    score = (similarities.mean() + 1) / 2 * 100
    return score
```

**Pros**: No need for quality labels, learns from pro players
**Cons**: Doesn't explain WHY score is low

### 6.2 Approach 2: Rule-Based Analysis

**Concept**: Measure biomechanical metrics directly.

```python
def rule_based_scoring(skeleton_sequence):
    # Find impact frame (highest wrist velocity)
    wrist_velocity = compute_velocity(skeleton_sequence[:, WRIST_IDX])
    impact_frame = np.argmax(wrist_velocity)
    
    scores = {}
    feedback = []
    
    # Check elbow angle (ideal: 120° at swing)
    elbow_angle = compute_angle(
        skeleton_sequence[impact_frame, SHOULDER_IDX],
        skeleton_sequence[impact_frame, ELBOW_IDX],
        skeleton_sequence[impact_frame, WRIST_IDX]
    )
    if elbow_angle < 120:
        scores['elbow'] = elbow_angle / 120 * 100
        feedback.append(f"Elbow angle {elbow_angle:.0f}° (ideal: 120°)")
    else:
        scores['elbow'] = 100
    
    # More metrics: shoulder rotation, wrist speed, etc.
    ...
    
    return {
        'total_score': np.mean(list(scores.values())),
        'breakdown': scores,
        'feedback': feedback
    }
```

**Pros**: Explainable, actionable feedback
**Cons**: Requires biomechanics knowledge, may miss subtle patterns

### 6.3 Hybrid Approach (Best)

Combine both:
- **Deep Learning Score**: Overall similarity to pro (0-100)
- **Rule-Based Analysis**: Specific feedback on what to improve

---

## 7. Evaluation Strategy

### 7.1 Classification Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Accuracy** | Overall correct predictions | ≥ 85% |
| **F1-Score (Macro)** | Balanced across all classes | ≥ 0.82 |
| **Confusion Matrix** | Per-class error analysis | Visual inspection |

### 7.2 Quality Scoring Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Similarity Score** | Cosine similarity with pro reference | ≥ 0.7 |
| **Inter-rater Correlation** | If human labels available | ≥ 0.6 |

### 7.3 Baseline Comparison

| Model | Description | Expected Accuracy |
|-------|-------------|-------------------|
| **LSTM Baseline** | Flatten skeleton → LSTM → FC | ~78% |
| **GCN (no Attention)** | GCN → Average Pool → FC | ~82% |
| **ST-GCN** | Literature baseline | ~84% |
| **Ours (GCN + Attention)** | Proposed method | ~87% |

### 7.4 Ablation Study

| Configuration | Purpose |
|---------------|---------|
| Full Model | Baseline |
| - Temporal Attention | Show Attention contribution |
| - GCN (use MLP) | Show GCN contribution |
| - Both | LSTM baseline |

---

## 8. Development Workflow

### 8.1 Iterative Prototype Approach

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DEVELOPMENT PHILOSOPHY                          │
│                                                                     │
│  "Prototype Fast → Learn → Rebuild Better"                         │
│                                                                     │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐          │
│  │ Prototype   │ ──► │ Identify    │ ──► │ Rebuild     │          │
│  │ (Quick &    │     │ Failures &  │     │ (Clean &    │          │
│  │  Dirty)     │     │ Learnings   │     │  Robust)    │          │
│  └─────────────┘     └─────────────┘     └─────────────┘          │
│        │                    │                   │                   │
│        ▼                    ▼                   ▼                   │
│  [prototype_v1/]    [LESSONS_LEARNED]    [final_project/]          │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.2 Project Directory Structure

```
badminton-stroke-analysis/
├── docs/
│   └── topic.md                    # This document (anchor)
├── prototypes/
│   ├── v1_basic_lstm/             # First prototype
│   ├── v2_add_gcn/                # Second prototype
│   └── v3_full_pipeline/          # Third prototype
├── final/
│   ├── src/
│   │   ├── data/                  # Data loading & preprocessing
│   │   ├── models/                # Model definitions
│   │   ├── training/              # Training loops
│   │   └── evaluation/            # Metrics & visualization
│   ├── notebooks/                 # Experiments & analysis
│   ├── webapp/                    # Streamlit demo
│   └── tests/                     # Unit tests
├── data/
│   ├── raw/                       # Downloaded datasets
│   └── processed/                 # Preprocessed data
├── checkpoints/                   # Saved models
└── requirements.txt
```

---

## 9. Prototype Phases

### Phase 1: Proof of Concept (Days 1-3)

**Goal**: Verify data pipeline works end-to-end.

```
Prototype 1: Basic LSTM
─────────────────────────
✓ Load Kaggle dataset
✓ Simple preprocessing
✓ LSTM classifier
✓ Get ANY accuracy (even 40% is fine)

Success Criteria: Model trains and predicts something
```

**Folder**: `prototypes/v1_basic_lstm/`

**Key Questions to Answer**:
- [ ] What format is the data in?
- [ ] How many samples per class?
- [ ] What preprocessing is needed?

### Phase 2: Core Model (Days 4-7)

**Goal**: Implement GCN and verify improvement over LSTM.

```
Prototype 2: Add GCN
─────────────────────────
✓ Implement simple GCN layer
✓ Build skeleton adjacency matrix
✓ Replace LSTM with GCN
✓ Compare accuracy with Phase 1

Success Criteria: GCN accuracy > LSTM accuracy
```

**Folder**: `prototypes/v2_add_gcn/`

**Key Questions to Answer**:
- [ ] Does GCN improve over LSTM?
- [ ] What adjacency matrix works best?
- [ ] How many GCN layers?

### Phase 3: Temporal Modeling (Days 8-12)

**Goal**: Add Temporal Attention module.

```
Prototype 3: Add Attention
─────────────────────────
✓ Implement Temporal Attention
✓ Combine with GCN
✓ Visualize attention weights
✓ Run ablation study

Success Criteria: Full model > GCN alone
```

**Folder**: `prototypes/v3_full_pipeline/`

### Phase 4: Quality Scoring (Days 13-16)

**Goal**: Add quality assessment (if time permits).

```
Prototype 4: Quality Scoring
─────────────────────────
✓ Implement similarity-based scoring
✓ Add rule-based analysis
✓ Generate feedback text

Success Criteria: Meaningful quality differentiation
```

### Phase 5: Final Integration (Days 17-21)

**Goal**: Clean code, build demo, write report.

```
Final: Production-Ready
─────────────────────────
✓ Refactor code from prototypes
✓ Build Streamlit webapp
✓ Write documentation
✓ Prepare presentation

Success Criteria: Demo works, report complete
```

**Folder**: `final/`

---

## 10. Technical Specifications

### 10.1 Development Environment

```bash
# Python version
python >= 3.9

# Core dependencies
torch >= 2.0
torch-geometric >= 2.3
mediapipe >= 0.10
numpy >= 1.24
pandas >= 2.0
streamlit >= 1.28

# Development tools
jupyter
matplotlib
seaborn
tqdm
```

### 10.2 Hardware Optimization (GTX 1650 - 4GB VRAM)

```python
# Mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# Small batch size
batch_size = 8  # or 4 if OOM

# Gradient checkpointing (if needed)
model.gradient_checkpointing_enable()
```

### 10.3 Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Sequence Length** | 64 frames | Pad/truncate to this |
| **GCN Hidden Dim** | 128 | Balance speed/accuracy |
| **GCN Layers** | 3 | More may oversmooth |
| **Attention Heads** | 4 | Standard for small models |
| **Learning Rate** | 1e-3 | Adam optimizer |
| **Batch Size** | 8 | Limited by GPU |
| **Epochs** | 100 | With early stopping |

---

## 11. Risk Mitigation & Fallback Plans

### 11.1 Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Dataset too small | Medium | High | Augmentation, pre-training on NTU |
| Model overfits | Medium | Medium | Dropout, early stopping, regularization |
| GCN doesn't help | Low | High | Fall back to LSTM + Attention |
| Quality scoring fails | Medium | Medium | Focus on classification only |
| GPU memory issues | Medium | Medium | Smaller batch, mixed precision |
| Time overrun | Medium | High | Cut quality scoring, simplify demo |

### 11.2 Fallback Plan

If full project is too ambitious:

| Component | Full Version | Fallback |
|-----------|--------------|----------|
| **Output** | Classification + Quality Score | Classification only |
| **Model** | GCN + Attention | GCN only (or LSTM + Attention) |
| **Demo** | Full web app | Jupyter notebook demo |
| **Baseline Comparison** | 4 models | 2 models |

**Minimum Viable Project**:
- Stroke classification with GCN
- Accuracy ≥ 80%
- Simple demo
- Basic report

---

## 12. References & Resources

### 12.1 Essential Papers

| Paper | Year | Relevance |
|-------|------|-----------|
| [ST-GCN](https://arxiv.org/abs/1801.07455) | 2018 | Core architecture reference |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 2017 | Attention mechanism |
| [ShuttleSet](https://arxiv.org/abs/2306.04948) | 2023 | Badminton dataset |
| [BST Transformer](https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer) | 2024 | Badminton + skeleton |

### 12.2 Code Repositories

| Repository | URL | Use |
|------------|-----|-----|
| MMAction2 | https://github.com/open-mmlab/mmaction2 | ST-GCN implementation |
| PyTorch Geometric | https://pytorch-geometric.readthedocs.io/ | GCN implementation |
| MediaPipe | https://mediapipe.dev/ | Pose estimation |
| CoachAI Projects | https://github.com/wywyWang/CoachAI-Projects | ShuttleSet dataset |

### 12.3 Tutorials

| Topic | Resource |
|-------|----------|
| GCN Explained | [Distill.pub: GNN Intro](https://distill.pub/2021/gnn-intro/) |
| Attention Explained | [Jay Alammar's Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) |
| ST-GCN Tutorial | YouTube: "ST-GCN Explained" |
| PyTorch Geometric | [Official Tutorials](https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html) |

---

## 13. Lessons Learned Log

> **Purpose**: Document learnings from each prototype iteration. Update this section as you progress.

### Prototype 1: Basic LSTM

| Date | Learning | Action Taken |
|------|----------|--------------|
| _TBD_ | _What went wrong/right_ | _How you fixed/used it_ |

### Prototype 2: Add GCN

| Date | Learning | Action Taken |
|------|----------|--------------|
| _TBD_ | _What went wrong/right_ | _How you fixed/used it_ |

### Prototype 3: Full Pipeline

| Date | Learning | Action Taken |
|------|----------|--------------|
| _TBD_ | _What went wrong/right_ | _How you fixed/used it_ |

### General Insights

| Category | Insight |
|----------|---------|
| **Data** | _Insights about data quality, preprocessing_ |
| **Model** | _What architectures worked/didn't work_ |
| **Training** | _Hyperparameter insights, optimization tips_ |
| **Debugging** | _Common errors and fixes_ |

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-21 | 1.0 | Initial document creation |
| 2026-01-28 | 2.0 | Major restructure: added prototype workflow, comprehensive technical specs, lessons learned log |

---

*This document is the authoritative reference for the Badminton Stroke Quality Assessment project. Update it as the project evolves.*
