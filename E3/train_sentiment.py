from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, random_split

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common.utils import count_parameters, ensure_dir, get_device, save_json, set_seed
from E3.text_data import (
    TOY_SENTIMENT_SAMPLES,
    SentimentDataset,
    build_vocab,
    collate_sentiment_batch,
)
from E3.text_models import SentimentRNNClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small packed-sequence sentiment RNN.")
    parser.add_argument("--rnn-type", choices=["lstm", "gru"], default="lstm")
    parser.add_argument("--output-dir", type=str, default="E3/results/sentiment")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--no-bidirectional", action="store_true")
    return parser.parse_args()


def evaluate_sentiment(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for tokens, lengths, labels in loader:
            tokens = tokens.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            logits = model(tokens, lengths)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    return {
        "loss": total_loss / len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = ensure_dir(args.output_dir)
    device = get_device(args.device)

    vocab = build_vocab(TOY_SENTIMENT_SAMPLES)
    dataset = SentimentDataset(TOY_SENTIMENT_SAMPLES, vocab)
    train_size = int(len(dataset) * 0.75)
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)
    collate = functools.partial(collate_sentiment_batch, pad_index=vocab.pad_index)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = SentimentRNNClassifier(
        vocab_size=len(vocab.itos),
        pad_index=vocab.pad_index,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        rnn_type=args.rnn_type,
        bidirectional=not args.no_bidirectional,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        for tokens, lengths, labels in train_loader:
            tokens = tokens.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total_items += labels.size(0)

        val = evaluate_sentiment(model, val_loader, criterion, device)
        row = {
            "epoch": float(epoch),
            "train_loss": total_loss / total_items,
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "val_macro_f1": val["macro_f1"],
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}/{args.epochs} | "
            f"train loss {row['train_loss']:.4f} | "
            f"val acc {row['val_accuracy']:.4f}, macro F1 {row['val_macro_f1']:.4f}"
        )

    payload = {
        "task": "toy_sentiment_analysis",
        "rnn_type": args.rnn_type,
        "bidirectional": not args.no_bidirectional,
        "vocab_size": len(vocab.itos),
        "parameters": count_parameters(model),
        "history": history,
    }
    save_json(payload, out_dir / f"sentiment_{args.rnn_type}.json")
    print(f"Saved sentiment result to: {out_dir}")


if __name__ == "__main__":
    main()

