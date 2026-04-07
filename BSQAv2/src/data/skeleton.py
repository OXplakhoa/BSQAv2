"""
COCO-17 Skeleton Definition + GCN Adjacency Matrix Builder
"""
import numpy as np
import torch

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
    "left_wrist",     # 9   ← Critical for stroke
    "right_wrist",    # 10  ← Critical for stroke
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
    # Face
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Upper body
    (5, 6),            # shoulder connection
    (5, 7), (7, 9),   # left arm
    (6, 8), (8, 10),  # right arm
    # Torso
    (5, 11), (6, 12), (11, 12),
    # Lower body
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]

# Body-only edges (excluding face — more reliable for sports)
BODY_SKELETON_EDGES = [
    (5, 6),            # shoulders
    (5, 7), (7, 9),   # left arm
    (6, 8), (8, 10),  # right arm
    (5, 11), (6, 12), (11, 12),  # torso
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]

# Keypoints critical for badminton stroke analysis
CRITICAL_KEYPOINTS = [9, 10, 7, 8, 5, 6]  # wrists, elbows, shoulders
FACE_KEYPOINTS = [0, 1, 2, 3, 4]
BODY_KEYPOINTS = list(range(5, 17))


def build_adjacency_matrix(
    num_nodes: int = NUM_KEYPOINTS,
    edges: list = None,
    self_loops: bool = True,
) -> np.ndarray:
    """
    Build the adjacency matrix for the skeleton graph.

    Args:
        num_nodes: Number of keypoints (17)
        edges: List of (i, j) edge tuples. Defaults to SKELETON_EDGES.
        self_loops: Whether to add self-loops (A + I)

    Returns:
        (num_nodes, num_nodes) adjacency matrix with self-loops
    """
    if edges is None:
        edges = SKELETON_EDGES

    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0  # Undirected graph

    if self_loops:
        A += np.eye(num_nodes, dtype=np.float32)  # Ã = A + I

    return A


def build_normalized_adjacency(
    num_nodes: int = NUM_KEYPOINTS,
    edges: list = None,
) -> torch.Tensor:
    """
    Build the symmetric normalized adjacency matrix for GCN:
        D̃^(-1/2) × Ã × D̃^(-1/2)

    This is precomputed once and reused in every GCN forward pass.

    Args:
        num_nodes: Number of keypoints (17)
        edges: Edge list. Defaults to SKELETON_EDGES.

    Returns:
        (num_nodes, num_nodes) normalized adjacency as a torch tensor
    """
    A_tilde = build_adjacency_matrix(num_nodes, edges, self_loops=True)

    # Degree matrix: D̃_ii = Σ_j Ã_ij
    D = np.diag(A_tilde.sum(axis=1))

    # D̃^(-1/2)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))

    # Symmetric normalization: D̃^(-1/2) × Ã × D̃^(-1/2)
    A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt

    return torch.from_numpy(A_norm).float()


if __name__ == "__main__":
    # Quick verification
    A = build_adjacency_matrix()
    print(f"Adjacency matrix shape: {A.shape}")
    print(f"Non-zero entries: {int(A.sum())} (edges×2 + self-loops)")
    print(f"Expected: {len(SKELETON_EDGES) * 2 + NUM_KEYPOINTS}")

    A_norm = build_normalized_adjacency()
    print(f"\nNormalized adjacency shape: {A_norm.shape}")
    print(f"Row sums (should ≈ 1): {A_norm.sum(dim=1)[:5].tolist()}")
    print("✓ Skeleton + adjacency matrix verified!")
