import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import torch


def plot_curves(train_losses, val_losses, train_accs, val_accs, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # ----- Loss curve -----
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_loss_curve.png"))
    plt.close()

    # ----- Accuracy curve -----
    plt.figure()
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_accuracy_curve.png"))
    plt.close()


def save_confusion_matrix(y_true, y_pred, classes, out_path: str):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i, format(cm[i, j], "d"),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(out_path)
    plt.close()


def save_sample_predictions(images, y_true, y_pred, classes, out_path: str, max_samples: int = 25):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    images = images[:max_samples]
    y_true = y_true[:max_samples]
    y_pred = y_pred[:max_samples]

    # de-normalize (MNIST)
    images = images * 0.3081 + 0.1307

    grid_size = int(np.ceil(np.sqrt(max_samples)))

    plt.figure(figsize=(10, 10))
    for i in range(max_samples):
        plt.subplot(grid_size, grid_size, i + 1)
        img = images[i][0].cpu().numpy()
        plt.imshow(img, cmap="gray")
        plt.title(f"T:{y_true[i].item()} / P:{y_pred[i].item()}", fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
