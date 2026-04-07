"""
PyTorch Dataset for Badminton Stroke Data
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Optional, List

from .preprocessing import preprocess_sequence
from ..config import (
    DATA_DIR, SEQUENCE_LENGTH, NUM_KEYPOINTS, COORD_DIM,
    CLASS_TO_IDX, STROKE_TYPES
)


class BadmintonDataset(Dataset):
    """
    Dataset for badminton stroke classification.
    
    Loads CSV files, groups by video ID, preprocesses, and returns
    (sequence, label) pairs.
    """
    
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        sequence_length: int = SEQUENCE_LENGTH,
        transform: Optional[callable] = None,
    ):
        """
        Args:
            data_dir: Path to directory containing CSV files
            sequence_length: Target sequence length after preprocessing
            transform: Optional transform to apply to sequences
        """
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Load all data
        self.samples: List[Tuple[np.ndarray, int]] = []
        self._load_data()
    
    def _load_data(self):
        """Load and preprocess all CSV files."""
        for stroke_type in STROKE_TYPES:
            csv_path = self.data_dir / f"{stroke_type}_v1.csv"
            if not csv_path.exists():
                print(f"Warning: {csv_path} not found, skipping.")
                continue
            
            df = pd.read_csv(csv_path)
            label = CLASS_TO_IDX[stroke_type]
            
            # Group by video ID
            for video_id, group in df.groupby('id'):
                # Extract keypoints (columns 3 onwards are kpt_X_x, kpt_X_y)
                keypoint_cols = [c for c in df.columns if c.startswith('kpt_')]
                keypoints_flat = group[keypoint_cols].values  # (T, 34)
                
                # Reshape to (T, 17, 2)
                T = keypoints_flat.shape[0]
                keypoints = keypoints_flat.reshape(T, NUM_KEYPOINTS, COORD_DIM)
                
                # Preprocess
                keypoints = preprocess_sequence(keypoints, self.sequence_length)
                
                self.samples.append((keypoints, label))
        
        print(f"Loaded {len(self.samples)} video sequences")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        keypoints, label = self.samples[idx]
        
        # Flatten keypoints: (T, 17, 2) -> (T, 34)
        keypoints_flat = keypoints.reshape(self.sequence_length, -1)
        
        # Convert to tensor
        x = torch.from_numpy(keypoints_flat).float()
        
        if self.transform:
            x = self.transform(x)
        
        return x, label


def create_data_loaders(
    batch_size: int = 8,
    train_split: float = 0.8,
    seed: int = 42,
    **kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation data loaders.
    
    Args:
        batch_size: Batch size
        train_split: Fraction of data for training
        seed: Random seed for reproducibility
        **kwargs: Additional args passed to BadmintonDataset
    
    Returns:
        (train_loader, val_loader)
    """
    from torch.utils.data import random_split, DataLoader
    
    dataset = BadmintonDataset(**kwargs)
    
    # Split
    n_train = int(len(dataset) * train_split)
    n_val = len(dataset) - n_train
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val], generator=generator
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    return train_loader, val_loader
