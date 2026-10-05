from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from sklearn.metrics import accuracy_score, f1_score

from common.data import build_mnist_loaders
from common.engine import collect_predictions, evaluate, fit_classifier
from common.plots import (
    plot_confusion_matrix,
    plot_learning_curves,
    plot_misclassified,
    save_history_csv,
    save_summary_csv,
)
from common.utils import ensure_dir, get_device, save_json, seconds_to_text, set_seed
from E1.models import build_model


MODEL_ORDER = ["softmax", "mlp", "cnn"]
CLASS_NAMES = [str(i) for i in range(10)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train E1 classifiers on MNIST.")
    parser.add_argument("--models", nargs="+", default=["all"], choices=["all", *MODEL_ORDER])
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="E1/results")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def selected_models(values: list[str]) -> list[str]:
    if "all" in values:
        return MODEL_ORDER
    return values


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = ensure_dir(args.output_dir)
    device = get_device(args.device)
    print(f"device: {device}")

    data = build_mnist_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    print(
        f"MNIST split: train={data.train_size}, val={data.val_size}, test={data.test_size}, "
        f"input_shape={data.input_shape}"
    )

    histories: dict[str, list[dict[str, float]]] = {}
    summary_rows: list[dict[str, object]] = []
    metrics_payload: dict[str, object] = {
        "dataset": "MNIST",
        "split": {
            "train": data.train_size,
            "validation": data.val_size,
            "test": data.test_size,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "device": str(device),
        },
        "models": {},
    }

    for model_name in selected_models(args.models):
        print(f"\n=== {model_name.upper()} ===")
        model = build_model(model_name)

        result = fit_classifier(
            model=model,
            train_loader=data.train_loader,
            val_loader=data.val_loader,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=device,
            show_progress=args.progress,
        )

        criterion = torch.nn.CrossEntropyLoss()
        test_metrics = evaluate(model, data.test_loader, criterion, device=device)
        y_true, y_pred, _probabilities, mistakes = collect_predictions(
            model=model,
            loader=data.test_loader,
            device=device,
            max_misclassified=25,
        )

        test_accuracy = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
        histories[model_name] = result.history

        save_history_csv(result.history, out_dir / f"history_{model_name}.csv")
        plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            class_names=CLASS_NAMES,
            path=out_dir / f"confusion_matrix_{model_name}.png",
            title=f"Confusion matrix - {model_name}",
        )
        plot_misclassified(
            mistakes=mistakes,
            class_names=CLASS_NAMES,
            path=out_dir / f"misclassified_{model_name}.png",
            title=f"Misclassified samples - {model_name}",
        )

        row = {
            "model": model_name,
            "parameters": result.parameter_count,
            "best_epoch": result.best_epoch,
            "best_val_accuracy": round(result.best_val_accuracy, 6),
            "test_loss": round(test_metrics.loss, 6),
            "test_accuracy": round(test_accuracy, 6),
            "macro_f1": round(macro_f1, 6),
            "training_time": seconds_to_text(result.training_time_seconds),
        }
        summary_rows.append(row)

        metrics_payload["models"][model_name] = {
            **row,
            "training_time_seconds": result.training_time_seconds,
        }
        print(
            f"test loss {test_metrics.loss:.4f}, "
            f"test acc {test_accuracy:.4f}, macro F1 {macro_f1:.4f}"
        )

    plot_learning_curves(histories, out_dir / "learning_curves.png")
    save_summary_csv(summary_rows, out_dir / "summary.csv")
    save_json(metrics_payload, out_dir / "metrics.json")

    print(f"\nSaved results to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()

