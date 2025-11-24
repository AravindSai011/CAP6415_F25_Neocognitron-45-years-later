# src/train.py

"""
Training script for the Neocognitron-inspired model on MNIST.


This script:
    - Loads MNIST (train/val/test) using dataset_loader.py
    - Trains the Neocognitron model
    - Tracks train/validation loss and accuracy
    - Saves the best model checkpoint
    - Saves training curves to results/training_curves.png
"""

import argparse
import os
from typing import Tuple, Dict, List

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset_loader import get_mnist_dataloaders
from src.models.neocognitron import Neocognitron


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Train Neocognitron-inspired model on MNIST.")

    parser.add_argument("--data_dir", type=str, default="data", help="Directory for MNIST data.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (cuda or cpu).",
    )
    parser.add_argument(
        "--save_dir", type=str, default="results/checkpoints", help="Directory to save model."
    )

    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train the model for one epoch.

    Args:
        model: Neocognitron model.
        dataloader: Training DataLoader.
        optimizer: Optimizer (e.g., Adam).
        device: torch.device('cuda') or torch.device('cpu').

    Returns:
        avg_loss, avg_acc
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in tqdm(dataloader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate the model on a validation or test set.

    Args:
        model: Trained model.
        dataloader: DataLoader for validation/test data.
        device: torch.device.

    Returns:
        avg_loss, avg_acc
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Eval", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def plot_curves(
    history: Dict[str, List[float]],
    save_path: str,
) -> None:
    """
    Plot training and validation curves.

    Args:
        history: Dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
        save_path: Path to save the PNG figure.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 4))

    # Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()

    # Accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curves")
    plt.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def main() -> None:
    """Main training routine."""
    args = parse_args()

    # Select device
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up data
    train_loader, val_loader, _ = get_mnist_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )

    # Initialize model
    model = Neocognitron(input_channels=1, num_classes=10).to(device)
    print(f"Model has {model.num_parameters():,} trainable parameters.")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Track history
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    # Prepare save directory
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_acc = 0.0
    best_ckpt_path = os.path.join(args.save_dir, "best_model.pt")

    # Training loop
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
            f"| Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"Saved new best model to {best_ckpt_path}")

    # Save curves
    plot_curves(history, save_path="results/training_curves.png")
    print("Training completed. Curves saved to results/training_curves.png")


if __name__ == "__main__":
    main()
