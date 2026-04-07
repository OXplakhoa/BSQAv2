"""
BSQAv2 Training Script (Skeleton)
Will support multi-model training and 5-Fold Cross Validation.
"""
import argparse
import sys
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="BSQAv2 Training Pipeline")
    parser.add_argument("--model", type=str, default="lstm_baseline", 
                        choices=["lstm_baseline", "gcn_bilstm", "gcn_bilstm_attn"],
                        help="Model architecture to train")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--augment", action="store_true", help="Apply data augmentation")
    parser.add_argument("--quick-test", action="store_true", help="Run 1 batch per epoch just to verify pipeline")
    parser.add_argument("--fold", type=int, default=None, help="Train specific fold only (0-4)")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Starting training for model: {args.model}")
    print("NOTE: Training loop implementation will be built in Phase 3.")

if __name__ == "__main__":
    main()
