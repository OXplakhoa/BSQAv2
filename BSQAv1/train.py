"""
Training script for BSQA LSTM Baseline
Usage: python train.py [--epochs N] [--quick-test]
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path

from src.data.dataset import create_data_loaders
from src.models.lstm_baseline import LSTMBaseline
from src.config import (
    NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE, TRAIN_SPLIT,
    CHECKPOINT_DIR, RUNS_DIR, IDX_TO_CLASS
)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        logits = model(x)
        loss = criterion(logits, y)
        
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    
    return total_loss / total, correct / total


def main(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data
    print("Loading data...")
    train_loader, val_loader = create_data_loaders(
        batch_size=args.batch_size,
        train_split=TRAIN_SPLIT,
    )
    
    # Model
    model = LSTMBaseline().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # TensorBoard
    RUNS_DIR.mkdir(exist_ok=True)
    writer = SummaryWriter(RUNS_DIR / "lstm_baseline")
    
    # Checkpoints
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    best_val_acc = 0
    epochs_without_improve = 0
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        # Logging
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
        
        print(f"Epoch {epoch:3d} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, CHECKPOINT_DIR / "best_model.pt")
            print(f"  → New best model saved! (val_acc: {val_acc:.4f})")
        else:
            epochs_without_improve += 1
        
        # Early stopping
        if epochs_without_improve >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    writer.close()
    
    # Final report
    print("\n" + "=" * 50)
    print(f"Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.1f}%)")
    target = 0.85
    if best_val_acc >= target:
        print(f"✓ Target accuracy ({target*100:.0f}%) ACHIEVED!")
    else:
        print(f"✗ Target accuracy ({target*100:.0f}%) not yet reached.")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BSQA LSTM Baseline")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--quick-test", action="store_true", help="Run 1 epoch only")
    
    args = parser.parse_args()
    
    if args.quick_test:
        args.epochs = 1
        print("Quick test mode: running 1 epoch only")
    
    main(args)
