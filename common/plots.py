from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def save_history_csv(history: list[dict[str, float]], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_summary_csv(rows: list[dict[str, object]], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_learning_curves(histories: dict[str, list[dict[str, float]]], path: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    for name, history in histories.items():
        epochs = [int(row["epoch"]) for row in history]
        axes[0].plot(epochs, [row["train_loss"] for row in history], linestyle="--", label=f"{name} train")
        axes[0].plot(epochs, [row["val_loss"] for row in history], label=f"{name} val")
        axes[1].plot(epochs, [row["train_accuracy"] for row in history], linestyle="--", label=f"{name} train")
        axes[1].plot(epochs, [row["val_accuracy"] for row in history], label=f"{name} val")

    axes[0].set_title("Loss curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross entropy loss")
    axes[0].grid(alpha=0.25)

    axes[1].set_title("Accuracy curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(alpha=0.25)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    path: str | Path,
    title: str,
) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7.4, 7.4))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_misclassified(
    mistakes: list[tuple[np.ndarray, int, int]],
    class_names: list[str],
    path: str | Path,
    title: str,
) -> None:
    if not mistakes:
        return

    cols = 5
    rows = int(np.ceil(len(mistakes) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.15))
    axes_array = np.atleast_1d(axes).ravel()

    for ax, (image, true_label, pred_label) in zip(axes_array, mistakes):
        if image.shape[0] == 1:
            shown = image[0] * 0.3081 + 0.1307
            ax.imshow(np.clip(shown, 0, 1), cmap="gray")
        else:
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            shown = np.transpose(image * std + mean, (1, 2, 0))
            ax.imshow(np.clip(shown, 0, 1))
        ax.set_title(f"true {class_names[true_label]} / pred {class_names[pred_label]}", fontsize=8)
        ax.axis("off")

    for ax in axes_array[len(mistakes) :]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
