from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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
from E1.models import build_model as build_e1_model
from E3.models import build_image_sequence_model


SEQUENCE_EXPERIMENTS = ["lstm_row", "gru_row", "lstm_patch", "gru_patch"]
BASELINE_EXPERIMENTS = ["mlp_baseline", "cnn_baseline"]
EXPERIMENTS = [*SEQUENCE_EXPERIMENTS, *BASELINE_EXPERIMENTS]
CLASS_NAMES = [str(i) for i in range(10)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train E3 LSTM/GRU classifiers on MNIST.")
    parser.add_argument("--experiments", nargs="+", default=["all"], choices=["all", *EXPERIMENTS])
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="E3/results")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=7)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def selected_experiments(values: list[str]) -> list[str]:
    if "all" in values:
        return EXPERIMENTS
    return values


def build_experiment_model(name: str, args: argparse.Namespace) -> torch.nn.Module:
    if name in SEQUENCE_EXPERIMENTS:
        return build_image_sequence_model(
            experiment=name,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            patch_size=args.patch_size,
            dropout=args.dropout,
            bidirectional=args.bidirectional,
        )
    if name == "mlp_baseline":
        return build_e1_model("mlp")
    if name == "cnn_baseline":
        return build_e1_model("cnn")
    raise ValueError(f"Unknown experiment: {name}")


def experiment_description(name: str) -> dict[str, str]:
    mapping = {
        "lstm_row": {"family": "LSTM", "sequence": "row", "purpose": "LSTM với mỗi hàng là một timestep"},
        "gru_row": {"family": "GRU", "sequence": "row", "purpose": "GRU với mỗi hàng là một timestep"},
        "lstm_patch": {"family": "LSTM", "sequence": "patch_7x7", "purpose": "LSTM với chuỗi patch 7x7"},
        "gru_patch": {"family": "GRU", "sequence": "patch_7x7", "purpose": "GRU với chuỗi patch 7x7"},
        "mlp_baseline": {"family": "MLP", "sequence": "flatten", "purpose": "Baseline không hồi quy từ E1"},
        "cnn_baseline": {"family": "CNN", "sequence": "spatial", "purpose": "Baseline không hồi quy từ E1"},
    }
    return mapping[name]


def write_combined_comparison(e3_rows: list[dict[str, object]], out_dir: Path) -> None:
    combined: list[dict[str, object]] = []
    for source, path in [
        ("E1", Path("E1/results/summary.csv")),
        ("E2", Path("E2/results/summary.csv")),
    ]:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["source"] = source
                combined.append(row)

    for row in e3_rows:
        row = dict(row)
        row["source"] = "E3"
        combined.append(row)

    if not combined:
        return

    keys: list[str] = []
    for row in combined:
        for key in row.keys():
            if key not in keys:
                keys.append(key)

    with (out_dir / "comparison_e1_e2_e3.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(combined)


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
        "rnn_config": {
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "patch_size": args.patch_size,
            "dropout": args.dropout,
            "bidirectional": args.bidirectional,
            "readout": "last timestep",
        },
        "experiments": {},
    }

    for experiment in selected_experiments(args.experiments):
        print(f"\n=== {experiment.upper()} ===")
        model = build_experiment_model(experiment, args)

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
        histories[experiment] = result.history

        save_history_csv(result.history, out_dir / f"history_{experiment}.csv")
        plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            class_names=CLASS_NAMES,
            path=out_dir / f"confusion_matrix_{experiment}.png",
            title=f"Confusion matrix - {experiment}",
        )
        plot_misclassified(
            mistakes=mistakes,
            class_names=CLASS_NAMES,
            path=out_dir / f"misclassified_{experiment}.png",
            title=f"Misclassified samples - {experiment}",
        )

        description = experiment_description(experiment)
        row = {
            "experiment": experiment,
            "family": description["family"],
            "sequence": description["sequence"],
            "parameters": result.parameter_count,
            "best_epoch": result.best_epoch,
            "best_val_accuracy": round(result.best_val_accuracy, 6),
            "test_loss": round(test_metrics.loss, 6),
            "test_accuracy": round(test_accuracy, 6),
            "macro_f1": round(macro_f1, 6),
            "training_time": seconds_to_text(result.training_time_seconds),
        }
        summary_rows.append(row)
        metrics_payload["experiments"][experiment] = {
            **row,
            "purpose": description["purpose"],
            "training_time_seconds": result.training_time_seconds,
        }
        print(
            f"test loss {test_metrics.loss:.4f}, "
            f"test acc {test_accuracy:.4f}, macro F1 {macro_f1:.4f}"
        )

    plot_learning_curves(histories, out_dir / "learning_curves.png")
    save_summary_csv(summary_rows, out_dir / "summary.csv")
    write_combined_comparison(summary_rows, out_dir)
    save_json(metrics_payload, out_dir / "metrics.json")

    print(f"\nSaved results to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()

