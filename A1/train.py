from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from A1.data import build_cifar100_subset_loaders
from A1.models import build_a1_model, count_parameters, parse_experiment_name
from common.engine import collect_predictions, evaluate, fit_classifier
from common.plots import (
    plot_confusion_matrix,
    plot_learning_curves,
    plot_misclassified,
    save_history_csv,
    save_summary_csv,
)
from common.utils import ensure_dir, get_device, save_json, seconds_to_text, set_seed


EXPERIMENTS = [
    "resnet18_scratch",
    "resnet18_pretrained_head",
    "resnet18_pretrained_partial",
    "resnet18_pretrained_full",
    "vit_b16_scratch",
    "vit_b16_pretrained_head",
    "vit_b16_pretrained_partial",
    "vit_b16_pretrained_full",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train A1 CNN vs Transformer experiments.")
    parser.add_argument("--experiments", nargs="+", default=["all"], choices=["all", *EXPERIMENTS])
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="A1/results")
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--max-train-per-class", type=int, default=500)
    parser.add_argument("--max-test-per-class", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-download-weights", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def selected_experiments(values: list[str]) -> list[str]:
    if "all" in values:
        return EXPERIMENTS
    return values


def describe_experiment(experiment: str) -> dict[str, str]:
    architecture, mode = parse_experiment_name(experiment)
    family = "CNN" if architecture == "resnet18" else "Transformer"
    readable_mode = {
        "scratch": "from_scratch",
        "pretrained_head": "pretrained_freeze_backbone_train_head",
        "pretrained_partial": "pretrained_freeze_partial",
        "pretrained_full": "pretrained_full_finetune",
    }[mode]
    return {
        "architecture": architecture,
        "family": family,
        "mode": readable_mode,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = ensure_dir(args.output_dir)
    device = get_device(args.device)
    print(f"device: {device}")

    data = build_cifar100_subset_loaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        num_classes=args.num_classes,
        max_train_per_class=args.max_train_per_class,
        max_test_per_class=args.max_test_per_class,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        download=True,
    )
    print(
        f"CIFAR-100 subset: classes={len(data.class_names)}, train={data.train_size}, "
        f"val={data.val_size}, test={data.test_size}, image_size={data.image_size}"
    )

    histories: dict[str, list[dict[str, float]]] = {}
    summary_rows: list[dict[str, object]] = []
    metrics_payload: dict[str, object] = {
        "dataset": "CIFAR-100 subset",
        "class_names": data.class_names,
        "split": {
            "train": data.train_size,
            "validation": data.val_size,
            "test": data.test_size,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "num_classes": args.num_classes,
            "max_train_per_class": args.max_train_per_class,
            "max_test_per_class": args.max_test_per_class,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "device": str(device),
        },
        "experiments": {},
    }

    for experiment in selected_experiments(args.experiments):
        print(f"\n=== {experiment.upper()} ===")
        architecture, mode = parse_experiment_name(experiment)
        model = build_a1_model(
            architecture=architecture,
            mode=mode,
            num_classes=len(data.class_names),
            download_weights=not args.no_download_weights,
        )
        total_params, trainable_params = count_parameters(model)

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
            class_names=data.class_names,
            path=out_dir / f"confusion_matrix_{experiment}.png",
            title=f"Confusion matrix - {experiment}",
        )
        plot_misclassified(
            mistakes=mistakes,
            class_names=data.class_names,
            path=out_dir / f"misclassified_{experiment}.png",
            title=f"Misclassified samples - {experiment}",
        )

        description = describe_experiment(experiment)
        row = {
            "experiment": experiment,
            "family": description["family"],
            "architecture": description["architecture"],
            "mode": description["mode"],
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
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
            "training_time_seconds": result.training_time_seconds,
        }
        print(
            f"test loss {test_metrics.loss:.4f}, "
            f"test acc {test_accuracy:.4f}, macro F1 {macro_f1:.4f}"
        )

    plot_learning_curves(histories, out_dir / "learning_curves.png")
    save_summary_csv(summary_rows, out_dir / "summary.csv")
    save_json(metrics_payload, out_dir / "metrics.json")
    print(f"\nSaved results to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
