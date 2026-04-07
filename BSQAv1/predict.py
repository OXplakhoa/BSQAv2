# Input: File CSV sample - took from @motion-data-for-badminton-shots
# Output: "The Stroke is: {name} - {confidence}"

# 1. Setup
# 2. Load Model
# 3. Load & Process Data 
# 4. Forward Pass
# 5. Post-processing

import argparse
import numpy as np
import pandas as pd
import torch

from src.models.lstm_baseline import LSTMBaseline
from src.config import (
    NUM_KEYPOINTS, COORD_DIM, SEQUENCE_LENGTH,
    CLASS_TO_IDX, IDX_TO_CLASS, CHECKPOINT_DIR
)
from src.data.preprocessing import preprocess_sequence


def predict_single(model, keypoints: np.ndarray, device: torch.device) -> dict:
    """
    Predict stroke type for a single sequence of keypoints.

    Args:
        model: Trained LSTMBaseline model (in eval mode)
        keypoints: (T, 17, 2) raw keypoint array
        device: torch device

    Returns:
        dict with 'stroke', 'confidence', and 'probabilities'
    """
    # Preprocess
    keypoints = preprocess_sequence(keypoints, SEQUENCE_LENGTH)

    # Flatten: (T, 17, 2) -> (T, 34)
    keypoints_flat = keypoints.reshape(SEQUENCE_LENGTH, -1)

    # Convert to tensor: (1, T, 34)
    x = torch.from_numpy(keypoints_flat).float().unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1)[0]
        predicted_idx = probs.argmax().item()
        confidence = probs[predicted_idx].item()

    return {
        'stroke': IDX_TO_CLASS[predicted_idx],
        'confidence': confidence,
        'probabilities': {IDX_TO_CLASS[i]: probs[i].item() for i in range(len(IDX_TO_CLASS))},
    }


def load_model(model_path: str, device: torch.device) -> LSTMBaseline:
    """Load a trained model from checkpoint."""
    model = LSTMBaseline().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    # Support both full checkpoint dict and raw state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', '?')} "
              f"(val_acc: {checkpoint.get('val_acc', '?'):.4f})")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model state dict.")

    model.eval()
    return model


def predict_csv(model, csv_path: str, device: torch.device):
    """
    Predict stroke type for all video sequences in a CSV file.

    Args:
        model: Trained model
        csv_path: Path to CSV with keypoint data
        device: torch device
    """
    df = pd.read_csv(csv_path)
    keypoint_cols = [c for c in df.columns if c.startswith('kpt_')]

    print(f"\nPredicting from: {csv_path}")
    print(f"Found {df['id'].nunique()} video sequences\n")

    for video_id, group in df.groupby('id'):
        keypoints_flat = group[keypoint_cols].values  # (T, 34)

        # Reshape to (T, 17, 2)
        T = keypoints_flat.shape[0]
        keypoints = keypoints_flat.reshape(T, NUM_KEYPOINTS, COORD_DIM)

        result = predict_single(model, keypoints, device)

        print(f"Video {video_id:>4} | "
              f"Stroke: {result['stroke']:>8} | "
              f"Confidence: {result['confidence']:.4f} | "
              f"Probs: {result['probabilities']}")


def main():
    parser = argparse.ArgumentParser(description="BSQA Stroke Prediction")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input CSV file with keypoint data")
    parser.add_argument("--model", type=str,
                        default=str(CHECKPOINT_DIR / "best_model.pt"),
                        help="Path to model checkpoint")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = load_model(args.model, device)

    # Predict
    predict_csv(model, args.input, device)


if __name__ == "__main__":
    main()
