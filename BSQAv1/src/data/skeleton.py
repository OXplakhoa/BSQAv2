"""
COCO-17 Skeleton Definition for Badminton Pose Data
"""

NUM_KEYPOINTS = 17

KEYPOINT_NAMES = [
    "nose",           # 0
    "left_eye",       # 1
    "right_eye",      # 2
    "left_ear",       # 3
    "right_ear",      # 4
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9  - Critical for stroke
    "right_wrist",    # 10 - Critical for stroke
    "left_hip",       # 11
    "right_hip",      # 12
    "left_knee",      # 13
    "right_knee",     # 14
    "left_ankle",     # 15
    "right_ankle",    # 16
]

KEYPOINT_TO_IDX = {name: idx for idx, name in enumerate(KEYPOINT_NAMES)}

# Skeleton edges (bone connections)
SKELETON_EDGES = [
    # Face (often missing in dataset)
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Upper body
    (5, 6),   # shoulder connection
    (5, 7), (7, 9),   # left arm
    (6, 8), (8, 10),  # right arm
    # Torso
    (5, 11), (6, 12), (11, 12),
    # Lower body
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]

# Body-only edges (excluding face, more reliable)
BODY_SKELETON_EDGES = [
    (5, 6),   # shoulders
    (5, 7), (7, 9),   # left arm
    (6, 8), (8, 10),  # right arm
    (5, 11), (6, 12), (11, 12),  # torso
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]

# Keypoints critical for badminton stroke analysis
CRITICAL_KEYPOINTS = [
    9, 10,  # wrists (racket hand)
    7, 8,   # elbows
    5, 6,   # shoulders
]

# Face keypoints (often missing, may filter out)
FACE_KEYPOINTS = [0, 1, 2, 3, 4]
BODY_KEYPOINTS = list(range(5, 17))
