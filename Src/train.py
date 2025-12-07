import os
import torch
import torch.nn as nn
import torch.optim as optim

from model import LateralInhibitionNeocognitron
from dataset import get_dataloaders
from utils import plot_curves



# ======= CONFIG (change here if you want) =======
EPOCHS = 5
BATCH_SIZE = 64
LR = 1e-3
OUTPUT_DIR = "results"
# ===============================================


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return loss_sum / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss_sum += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    return loss_sum / total, correct / total


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_loader, val_loader, _ = get_dataloaders(batch_size=BATCH_SIZE)

    model = LateralInhibitionNeocognitron().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_acc = 0.0

    for epoch in range(EPOCHS):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = validate(model, val_loader, criterion, device)

        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        train_accs.append(tr_acc)
        val_accs.append(va_acc)

        print(
            f"Epoch {epoch+1}/{EPOCHS} "
            f"- Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} "
            f"- Val Loss: {va_loss:.4f} Acc: {va_acc:.4f}"
        )

        if va_acc > best_acc:
            best_acc = va_acc
            ckpt_path = os.path.join(OUTPUT_DIR, "best_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Saved new best model to {ckpt_path}")

    plot_curves(train_losses, val_losses, train_accs, val_accs, OUTPUT_DIR)
    print("Training finished.")


if __name__ == "__main__":
    main()
