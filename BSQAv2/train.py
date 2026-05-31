"""
BSQAv2 Training Pipeline — Phase 3

Supports multi-model training with 5-Fold stratified CV,
class-weighted loss, mixed precision (AMP), early stopping,
ReduceLROnPlateau, and TensorBoard logging.
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

_project_root = Path(__file__).parent
sys.path.insert(0, str(_project_root))

from src.config import (
    BATCH_SIZE, NUM_CLASSES, LEARNING_RATE, NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE, K_FOLDS, SEED, CHECKPOINT_DIR, RUNS_DIR,
    STROKE_TYPES, CLASS_TO_IDX,
    NUM_KEYPOINTS, COORD_DIM,
)
from src.data.augmentation import random_augmentation
from src.data.dataset import create_kfold_loaders
from src.models.lstm_baseline import LSTMBaseline
from src.models.bilstm_baseline import BiLSTMBaseline
from src.models.gcn_bilstm import GCNBiLSTM
from src.models.gcn_bilstm_attn import GCNBiLSTMAttention
from src.utils.metrics import compute_metrics, aggregate_fold_metrics, print_fold_summary


# ─── Model Registry ────────────────────────────────────────────────────────────


def _json_safe(obj):
    """Recursively convert numpy types to Python native for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    return obj


# (factory_or_class, has_quality_head)


def _gcn_lstm_factory():
    return GCNBiLSTM(bidirectional=False)


MODEL_REGISTRY: Dict[str, Tuple] = {
    "lstm_baseline":       (LSTMBaseline,       False),
    "bilstm_baseline":     (BiLSTMBaseline,     False),
    "gcn_lstm":            (_gcn_lstm_factory,  False),
    "gcn_bilstm":          (GCNBiLSTM,          False),
    "gcn_bilstm_attn":     (GCNBiLSTMAttention, True),
}


def _build_model(name: str) -> nn.Module:
    factory, quality = MODEL_REGISTRY[name]
    model = factory()

    if quality:
        model = _DualHeadWrapper(model)

    return model


class _DualHeadWrapper(nn.Module):
    """Wraps a dual-head model; returns only classification logits."""
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _quality, _attn = self.inner(x)
        return logits


# ─── Device & AMP Setup ────────────────────────────────────────────────────────


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ─── Training Logic ────────────────────────────────────────────────────────────


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    quick_test: bool = False,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        if quick_test and batch_idx >= 1:
            break

    return total_loss / total, correct / total if total > 0 else 0.0


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    total = 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(y.cpu().numpy())
        total += y.size(0)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    metrics = compute_metrics(all_labels, all_preds, class_names=STROKE_TYPES)

    return total_loss / total, metrics


def train_fold(
    model_name: str,
    fold_idx: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    run_dir: Path,
    args: argparse.Namespace,
) -> Dict:
    model = _build_model(model_name).to(device)

    # ── compute class weights from training labels ──
    all_train_labels = []
    for _, y in train_loader:
        all_train_labels.append(y.numpy())
    all_train_labels = np.concatenate(all_train_labels)
    class_counts = np.bincount(all_train_labels, minlength=NUM_CLASSES)
    inv_freq = len(all_train_labels) / (NUM_CLASSES * (class_counts + 1e-8))
    class_weights = torch.tensor(inv_freq, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-6,
    )
    scaler = GradScaler() if device.type == "cuda" else None

    writer = SummaryWriter(log_dir=str(run_dir / f"fold_{fold_idx}"))
    best_val_loss = float("inf")
    patience_counter = 0
    best_metrics = None
    last_metrics = None

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            quick_test=args.quick_test,
        )
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        last_metrics = val_metrics

        if not np.isfinite(val_loss):
            print(f"  Epoch {epoch+1:3d}/{args.epochs} | "
                  f"train loss={train_loss:.4f} acc={train_acc:.3f} | "
                  f"val loss=NaN (skip) | "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")
            patience_counter += 1
        elif val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_metrics = val_metrics
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "val_loss": val_loss, "val_metrics": val_metrics},
                run_dir / f"best_model_fold{fold_idx}.pth",
            )
        else:
            patience_counter += 1

        print(f"  Epoch {epoch+1:3d}/{args.epochs} | "
              f"train loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"val loss={val_loss:.4f} acc={val_metrics['accuracy']:.3f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        # Log to TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/val", val_metrics["accuracy"], epoch)
        writer.add_scalar("F1/val_macro", val_metrics.get("f1_macro", 0), epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        if patience_counter >= args.patience:
            print(f"  --- Early stopping at epoch {epoch+1} ---")
            break

        if args.quick_test and epoch >= 2:
            break

    writer.close()
    return best_metrics if best_metrics is not None else last_metrics


# ─── Main ──────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="BSQAv2 Training Pipeline")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Model architecture to train")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader workers (set to 0 if issues)")
    parser.add_argument("--augment", action="store_true",
                        help="Apply data augmentation (train only)")
    parser.add_argument("--quick-test", action="store_true",
                        help="Minimal epochs/batches for pipeline verification")
    parser.add_argument("--fold", type=int, default=None,
                        help="Train single fold (0-4), default: all 5")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if args.device == "cuda" else
                          "cpu" if args.device == "cpu" else
                          get_device())

    print(f"Device: {device}")
    print(f"Model:  {args.model}")
    print(f"Augment: {args.augment} | Quick-Test: {args.quick_test}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.model}_{timestamp}"
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    augment_fn = random_augmentation if args.augment else None

    fold_loaders = create_kfold_loaders(
        k_folds=K_FOLDS,
        batch_size=args.batch_size,
        seed=SEED,
        augment_fn=augment_fn,
        num_workers=args.num_workers,
    )

    folds_to_run = [args.fold] if args.fold is not None else range(K_FOLDS)
    fold_metrics = []

    for fold_idx in folds_to_run:
        train_loader, val_loader = fold_loaders[fold_idx]
        print(f"\n{'='*60}")
        print(f"  Fold {fold_idx + 1}/{K_FOLDS} — Train={len(train_loader.dataset)}, "
              f"Val={len(val_loader.dataset)}")
        print(f"{'='*60}")

        metrics = train_fold(
            model_name=args.model,
            fold_idx=fold_idx,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            run_dir=run_dir,
            args=args,
        )
        fold_metrics.append(metrics)

    if len(fold_metrics) > 1:
        print_fold_summary(fold_metrics, model_name=args.model)
        agg = aggregate_fold_metrics(fold_metrics)
        with open(run_dir / "cv_summary.json", "w") as f:
            json.dump(_json_safe(agg), f, indent=2)
        print(f"\nCV summary saved to {run_dir / 'cv_summary.json'}")

    print(f"\nTensorBoard:  tensorboard --logdir={run_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
