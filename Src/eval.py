import os
import torch

from model import LateralInhibitionNeocognitron

from dataset import get_dataloaders
from utils import save_confusion_matrix, save_sample_predictions


# ======= CONFIG =======
CHECKPOINT_PATH = "results/best_model.pth"
OUTPUT_DIR = "results"
# ======================


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    _, _, test_loader = get_dataloaders()

    model = ImprovedNeoCNN().to(device)
    print(f"Loading checkpoint from {CHECKPOINT_PATH}")
    state = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    all_preds = []
    all_labels = []
    all_imgs = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = outputs.max(1)

            all_imgs.append(imgs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_imgs = torch.cat(all_imgs)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    save_confusion_matrix(
        all_labels.numpy(),
        all_preds.numpy(),
        classes=[str(i) for i in range(10)],
        out_path=os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
    )

    save_sample_predictions(
        all_imgs,
        all_labels,
        all_preds,
        classes=[str(i) for i in range(10)],
        out_path=os.path.join(OUTPUT_DIR, "sample_predictions.png"),
    )

    acc = (all_preds == all_labels).float().mean().item()
    print(f"Test Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
