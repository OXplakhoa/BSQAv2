"""
Utility to extract random test samples from the dataset
for quick model verification.

Usage:
    python -m src.utils.sample_extractor          # Extract & predict 2 samples per class
    python -m src.utils.sample_extractor --n 5    # Extract & predict 5 samples per class
"""
import argparse
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from ..config import (
    DATA_DIR, CHECKPOINT_DIR, STROKE_TYPES, CLASS_TO_IDX,
    NUM_KEYPOINTS, COORD_DIM, IDX_TO_CLASS
)
from ..data.preprocessing import preprocess_sequence
from ..models.lstm_baseline import LSTMBaseline


def extract_samples(n_per_class: int = 2, seed: int = 42) -> list:
    """
    Extract random video samples from each stroke type CSV.

    Args:
        n_per_class: Number of samples to extract per stroke type
        seed: Random seed for reproducibility

    Returns:
        List of dicts: {'stroke_type', 'video_id', 'keypoints', 'n_frames'}
    """
    random.seed(seed)
    samples = []

    for stroke_type in STROKE_TYPES:
        csv_path = DATA_DIR / f"{stroke_type}_v1.csv"
        if not csv_path.exists():
            print(f"⚠ {csv_path.name} not found, skipping.")
            continue

        df = pd.read_csv(csv_path)
        keypoint_cols = [c for c in df.columns if c.startswith('kpt_')]
        video_ids = df['id'].unique().tolist()

        # Pick random samples
        n = min(n_per_class, len(video_ids))
        selected_ids = random.sample(video_ids, n)

        for vid in selected_ids:
            group = df[df['id'] == vid]
            kpt_flat = group[keypoint_cols].values  # (T, 34)
            T = kpt_flat.shape[0]
            keypoints = kpt_flat.reshape(T, NUM_KEYPOINTS, COORD_DIM)

            samples.append({
                'stroke_type': stroke_type,
                'video_id': vid,
                'keypoints': keypoints,
                'n_frames': T,
            })

    return samples


def verify_model(model_path: str = None, n_per_class: int = 2, seed: int = 42):
    """
    Extract samples and run predictions to verify the model works correctly.

    Args:
        model_path: Path to checkpoint (defaults to best_model.pt)
        n_per_class: Number of test samples per stroke class
        seed: Random seed
    """
    if model_path is None:
        model_path = str(CHECKPOINT_DIR / "best_model.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = LSTMBaseline().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded (epoch {checkpoint.get('epoch', '?')}, "
              f"val_acc: {checkpoint.get('val_acc', 0):.4f})")
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Extract samples
    samples = extract_samples(n_per_class=n_per_class, seed=seed)
    print(f"\n{'='*70}")
    print(f"Verifying model with {len(samples)} samples "
          f"({n_per_class} per class)")
    print(f"{'='*70}\n")

    correct = 0
    total = 0
    results_by_class = {s: {'correct': 0, 'total': 0} for s in STROKE_TYPES}

    for sample in samples:
        true_label = sample['stroke_type']
        keypoints = sample['keypoints']

        # Preprocess
        processed = preprocess_sequence(keypoints, target_length=64)
        flat = processed.reshape(64, -1)
        x = torch.from_numpy(flat).float().unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            output = model(x)
            probs = torch.softmax(output, dim=1)[0]
            pred_idx = probs.argmax().item()
            pred_label = IDX_TO_CLASS[pred_idx]
            confidence = probs[pred_idx].item()

        is_correct = pred_label == true_label
        icon = "✓" if is_correct else "✗"

        print(f"  {icon} Video {sample['video_id']:>4} | "
              f"True: {true_label:>8} | "
              f"Pred: {pred_label:>8} ({confidence:.2%}) | "
              f"Frames: {sample['n_frames']}")

        correct += int(is_correct)
        total += 1
        results_by_class[true_label]['total'] += 1
        results_by_class[true_label]['correct'] += int(is_correct)

    # Summary
    print(f"\n{'='*70}")
    print(f"VERIFICATION SUMMARY")
    print(f"{'='*70}")
    print(f"Overall Accuracy: {correct}/{total} ({correct/total:.1%})")
    print()
    for stroke, stats in results_by_class.items():
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total']
            print(f"  {stroke:>8}: {stats['correct']}/{stats['total']} ({acc:.1%})")
    print(f"{'='*70}\n")

    return correct, total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify BSQA model with sample data")
    parser.add_argument("--n", type=int, default=2,
                        help="Number of samples per class (default: 2)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    verify_model(model_path=args.model, n_per_class=args.n, seed=args.seed)
