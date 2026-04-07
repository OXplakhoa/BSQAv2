"""
BSQA Configuration - Hyperparameters and settings
"""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "motion-data-for-badminton-shots"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RUNS_DIR = PROJECT_ROOT / "runs"

# Data
SEQUENCE_LENGTH = 64  # Pad/truncate to this many frames
NUM_KEYPOINTS = 17
COORD_DIM = 2  # x, y only (no z in this dataset)
INPUT_DIM = NUM_KEYPOINTS * COORD_DIM  # 34 features per frame

# Classes
STROKE_TYPES = ["smash", "lift", "net_shot"]
NUM_CLASSES = len(STROKE_TYPES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(STROKE_TYPES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(STROKE_TYPES)}

# Model
HIDDEN_DIM = 128
NUM_LSTM_LAYERS = 2
DROPOUT = 0.3

# Training
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
TRAIN_SPLIT = 0.8
EARLY_STOPPING_PATIENCE = 10

# Keypoint indices for normalization
HIP_LEFT_IDX = 11
HIP_RIGHT_IDX = 12
SHOULDER_LEFT_IDX = 5
SHOULDER_RIGHT_IDX = 6
