"""
PyTorch Dataset for Badminton Stroke Data — v2
Supports loading from multiple data directories (kaggle + youtube)
and 5-Fold stratified cross-validation with train-only augmentation.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
from typing import Tuple, Optional, List

from .preprocessing import preprocess_sequence
from ..config import (
    DATA_DIRS, SEQUENCE_LENGTH,
    NUM_KEYPOINTS, COORD_DIM, CLASS_TO_IDX, STROKE_TYPES,
    K_FOLDS, SEED, BATCH_SIZE
)


class BadmintonDataset(Dataset):
    """
    Dataset for badminton stroke classification.
    Loads CSV files from multiple directories, groups by video ID,
    preprocesses, and returns (sequence, label) pairs.

    Keeps data in (T, 17, 2) shape — NOT flattened.
    Flattening is the model's responsibility (LSTM flattens, GCN doesn't).
    """

    def __init__(
        self,
        data_dirs: Optional[List[Path]] = None,
        sequence_length: int = SEQUENCE_LENGTH,
        augment_fn: Optional[callable] = None,
    ):
        self.sequence_length = sequence_length
        self.augment_fn = augment_fn

        if data_dirs is None:
            data_dirs = DATA_DIRS

        self.samples: List[Tuple[np.ndarray, int]] = []
        self.labels: List[int] = []
        self._load_data(data_dirs)

    def _load_data(self, data_dirs: List[Path]):
        for data_dir in data_dirs:
            data_dir = Path(data_dir)
            if not data_dir.exists():
                continue

            for stroke_type in STROKE_TYPES:
                if stroke_type not in CLASS_TO_IDX:
                    continue
                label = CLASS_TO_IDX[stroke_type]

                # Load both v1 and v2 CSV files if present
                for version in ["v1", "v2"]:
                    csv_path = data_dir / f"{stroke_type}_{version}.csv"
                    if not csv_path.exists():
                        continue

                    df = pd.read_csv(csv_path)
                    keypoint_cols = [c for c in df.columns if c.startswith('kpt_')]

                    for _, group in df.groupby('id'):
                        keypoints_flat = group[keypoint_cols].values
                        T = keypoints_flat.shape[0]
                        keypoints = keypoints_flat.reshape(T, NUM_KEYPOINTS, COORD_DIM)

                        keypoints = preprocess_sequence(keypoints, self.sequence_length)
                        self.samples.append((keypoints, label))
                        self.labels.append(label)

        print(f"Loaded {len(self.samples)} video sequences "
              f"({len(set(self.labels))} classes)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        keypoints, label = self.samples[idx]

        # Apply augmentation if set (train-only)
        if self.augment_fn is not None:
            keypoints = self.augment_fn(keypoints.copy())

        # Return as (T, 17, 2) — model decides to flatten or use as graph
        x = torch.from_numpy(keypoints).float()
        return x, label


def create_kfold_loaders(
    k_folds: int = K_FOLDS,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
    augment_fn: Optional[callable] = None,
    data_dirs: Optional[List[Path]] = None,
    num_workers: int = 0,
) -> List[Tuple[DataLoader, DataLoader]]:
    """
    Create K-Fold stratified cross-validation data loaders.

    CRITICAL: Augmentation is applied to train split ONLY.
    Val data remains original and untouched.

    Args:
        k_folds: Number of folds (default: 5)
        batch_size: Batch size
        seed: Random seed for reproducibility
        augment_fn: Augmentation function (applied to train only)
        data_dirs: List of data directories to load from

    Returns:
        List of (train_loader, val_loader) tuples, one per fold
    """
    # Load full dataset WITHOUT augmentation
    full_dataset = BadmintonDataset(data_dirs=data_dirs, augment_fn=None)
    labels = np.array(full_dataset.labels)

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

    fold_loaders = []
    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(range(len(full_dataset)), labels)):
        # Train subset — WITH augmentation
        train_subset = Subset(full_dataset, train_indices)
        if augment_fn is not None:
            train_dataset = _AugmentedSubset(full_dataset, train_indices, augment_fn)
        else:
            train_dataset = train_subset

        # Val subset — NO augmentation (original data only)
        val_subset = Subset(full_dataset, val_indices)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        )

        print(f"Fold {fold_idx + 1}/{k_folds}: "
              f"Train={len(train_indices)}, Val={len(val_indices)}")

        fold_loaders.append((train_loader, val_loader))

    return fold_loaders


class _AugmentedSubset(Dataset):
    """Wraps a dataset subset and applies augmentation on-the-fly."""

    def __init__(self, dataset: BadmintonDataset, indices: np.ndarray, augment_fn: callable):
        self.dataset = dataset
        self.indices = indices
        self.augment_fn = augment_fn

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        keypoints, label = self.dataset.samples[real_idx]

        # Apply augmentation to a copy
        keypoints = self.augment_fn(keypoints.copy())
        x = torch.from_numpy(keypoints).float()
        return x, label
