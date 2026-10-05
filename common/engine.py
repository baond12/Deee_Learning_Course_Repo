from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from common.utils import count_parameters, now, seconds_to_text


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float


@dataclass
class TrainResult:
    history: list[dict[str, float]]
    best_epoch: int
    best_val_accuracy: float
    parameter_count: int
    training_time_seconds: float


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> tuple[int, int]:
    predictions = logits.argmax(dim=1)
    correct = (predictions == targets).sum().item()
    return correct, targets.numel()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    show_progress: bool,
) -> EpochMetrics:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    iterator = tqdm(loader, desc="train", leave=False, disable=not show_progress)
    for inputs, targets in iterator:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        correct, count = accuracy_from_logits(logits, targets)
        total_loss += loss.item() * batch_size
        total_correct += correct
        total_samples += count

    return EpochMetrics(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    show_progress: bool = False,
) -> EpochMetrics:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    iterator = tqdm(loader, desc="eval", leave=False, disable=not show_progress)
    for inputs, targets in iterator:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        correct, count = accuracy_from_logits(logits, targets)
        total_loss += loss.item() * batch_size
        total_correct += correct
        total_samples += count

    return EpochMetrics(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
    )


def fit_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    show_progress: bool,
) -> TrainResult:
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_accuracy = -1.0
    start_time = now()

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            show_progress=show_progress,
        )
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            show_progress=False,
        )

        row = {
            "epoch": float(epoch),
            "train_loss": train_metrics.loss,
            "train_accuracy": train_metrics.accuracy,
            "val_loss": val_metrics.loss,
            "val_accuracy": val_metrics.accuracy,
        }
        history.append(row)

        if val_metrics.accuracy > best_val_accuracy:
            best_val_accuracy = val_metrics.accuracy
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"epoch {epoch:02d}/{epochs} | "
            f"train loss {train_metrics.loss:.4f}, acc {train_metrics.accuracy:.4f} | "
            f"val loss {val_metrics.loss:.4f}, acc {val_metrics.accuracy:.4f}"
        )

    model.load_state_dict(best_state)
    training_time = now() - start_time
    print(f"best val acc {best_val_accuracy:.4f} at epoch {best_epoch}")
    print(f"training time: {seconds_to_text(training_time)}")

    return TrainResult(
        history=history,
        best_epoch=best_epoch,
        best_val_accuracy=best_val_accuracy,
        parameter_count=count_parameters(model),
        training_time_seconds=training_time,
    )


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_misclassified: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[np.ndarray, int, int]]]:
    model.eval()
    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    mistakes: list[tuple[np.ndarray, int, int]] = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

        all_targets.append(targets.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())
        all_probabilities.append(probabilities.cpu().numpy())

        wrong = predictions != targets
        for image, true_label, pred_label in zip(inputs[wrong], targets[wrong], predictions[wrong]):
            if len(mistakes) >= max_misclassified:
                break
            mistakes.append((image.cpu().numpy(), int(true_label.cpu()), int(pred_label.cpu())))

    return (
        np.concatenate(all_targets),
        np.concatenate(all_predictions),
        np.concatenate(all_probabilities),
        mistakes,
    )

