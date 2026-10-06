from __future__ import annotations

import functools
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from torch.utils.data import DataLoader

from E1.models import build_model as build_e1_model
from E2.models import build_model as build_e2_model
from E3.models import build_image_sequence_model
from E3.text_data import (
    TOY_SENTIMENT_SAMPLES,
    SentimentDataset,
    build_vocab,
    collate_sentiment_batch,
)
from E3.text_models import SentimentRNNClassifier


def assert_shape(name: str, actual: tuple[int, ...], expected: tuple[int, ...]) -> None:
    if actual != expected:
        raise AssertionError(f"{name}: expected {expected}, got {actual}")
    print(f"OK {name}: {actual}")


def main() -> None:
    torch.manual_seed(42)
    images = torch.randn(4, 1, 28, 28)

    for experiment in ["lstm_row", "gru_row", "lstm_patch", "gru_patch"]:
        model = build_image_sequence_model(experiment=experiment)
        logits = model(images)
        assert_shape(experiment, tuple(logits.shape), (4, 10))

    for baseline in ["mlp", "cnn"]:
        model = build_e1_model(baseline)
        logits = model(images)
        assert_shape(f"baseline_{baseline}", tuple(logits.shape), (4, 10))

    for experiment in ["custom_patch", "pytorch_patch", "custom_row"]:
        model = build_e2_model(experiment=experiment)
        logits = model(images)
        assert_shape(f"e2_{experiment}", tuple(logits.shape), (4, 10))

    vocab = build_vocab(TOY_SENTIMENT_SAMPLES)
    dataset = SentimentDataset(TOY_SENTIMENT_SAMPLES[:6], vocab)
    loader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=functools.partial(collate_sentiment_batch, pad_index=vocab.pad_index),
    )
    tokens, lengths, labels = next(iter(loader))
    assert_shape("sentiment_tokens", tuple(tokens.shape), (4, int(lengths.max().item())))
    assert_shape("sentiment_lengths", tuple(lengths.shape), (4,))
    assert_shape("sentiment_labels", tuple(labels.shape), (4,))

    for rnn_type in ["lstm", "gru"]:
        model = SentimentRNNClassifier(
            vocab_size=len(vocab.itos),
            pad_index=vocab.pad_index,
            rnn_type=rnn_type,
            bidirectional=True,
        )
        logits = model(tokens, lengths)
        assert_shape(f"sentiment_{rnn_type}", tuple(logits.shape), (4, 2))

    print("All E3 smoke tests passed.")


if __name__ == "__main__":
    main()

