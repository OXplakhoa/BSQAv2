"""
BSQAv2 Configuration — Hyperparameters and settings
"""
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR_KAGGLE = PROJECT_ROOT / "data" / "motion-data-for-badminton-shots"
DATA_DIR_YOUTUBE = PROJECT_ROOT / "data" / "youtube"
DATA_DIR_FRIEND = PROJECT_ROOT / "data" / "friend_data"
DATA_DIR_ARXIV = PROJECT_ROOT / "data" / "arxiv"
DATA_DIR_BADMINTON = PROJECT_ROOT / "data" / "badminton"

# All data directories loaded together by BadmintonDataset
DATA_DIRS = [DATA_DIR_KAGGLE, DATA_DIR_YOUTUBE, DATA_DIR_FRIEND, DATA_DIR_ARXIV, DATA_DIR_BADMINTON]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RUNS_DIR = PROJECT_ROOT / "runs"
RESULTS_DIR = PROJECT_ROOT / "results"

# ─── Data ─────────────────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 64       # Pad/truncate to this many frames
NUM_KEYPOINTS = 17         # COCO-17 format
COORD_DIM = 2              # x, y only
COORD_DIM_VELOCITY = 4      # x, y, dx, dy
INPUT_DIM = NUM_KEYPOINTS * COORD_DIM  # 34 features per frame (for LSTM baseline)
INPUT_DIM_VELOCITY = NUM_KEYPOINTS * COORD_DIM_VELOCITY  # 68 with velocity

# ─── Classes (5 stroke types) ────────────────────────────────────────────────
STROKE_TYPES = ["smash", "clear", "drop_shot", "net_shot", "lift"]
NUM_CLASSES = len(STROKE_TYPES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(STROKE_TYPES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(STROKE_TYPES)}

# ─── GCN ──────────────────────────────────────────────────────────────────────
GCN_HIDDEN_DIM = 128       # Output feature dim per joint
GCN_NUM_LAYERS = 3         # More may cause over-smoothing
GCN_POOL = "joint_attn"  # Learnable attention over 17 joints (works with LR=5e-4)

# ─── BiLSTM ───────────────────────────────────────────────────────────────────
BILSTM_HIDDEN_DIM = 128    # Per-direction hidden size (output = 256 bidirectional)
BILSTM_NUM_LAYERS = 2
BILSTM_DROPOUT = 0.2

# ─── Temporal Attention ───────────────────────────────────────────────────────
ATTENTION_HEADS = 4
ATTENTION_DIM = BILSTM_HIDDEN_DIM * 2  # 256 (bidirectional output)

# ─── LSTM Baseline (for ablation comparison with v1) ─────────────────────────
HIDDEN_DIM = 128
NUM_LSTM_LAYERS = 2
DROPOUT = 0.2

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE = 16
LEARNING_RATE = 5e-4
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
K_FOLDS = 5
SEED = 42

# ─── Keypoint indices for normalization (COCO-17) ────────────────────────────
HIP_LEFT_IDX = 11
HIP_RIGHT_IDX = 12
SHOULDER_LEFT_IDX = 5
SHOULDER_RIGHT_IDX = 6
