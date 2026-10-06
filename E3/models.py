from __future__ import annotations

import torch
from torch import nn


class RowSequenceTokenizer(nn.Module):
    sequence_length = 28
    feature_dim = 28

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(1)


class PatchSequenceTokenizer(nn.Module):
    def __init__(self, patch_size: int = 7) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.sequence_length = (28 // patch_size) * (28 // patch_size)
        self.feature_dim = patch_size * patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        patches = patches.contiguous().view(x.size(0), 1, -1, self.patch_size, self.patch_size)
        patches = patches.squeeze(1)
        return patches.flatten(2)


class ImageSequenceClassifier(nn.Module):
    def __init__(
        self,
        rnn_type: str = "lstm",
        sequence_type: str = "row",
        hidden_dim: int = 128,
        num_layers: int = 1,
        patch_size: int = 7,
        num_classes: int = 10,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        if sequence_type == "row":
            self.tokenizer = RowSequenceTokenizer()
        elif sequence_type == "patch":
            self.tokenizer = PatchSequenceTokenizer(patch_size=patch_size)
        else:
            raise ValueError(f"Unknown sequence_type: {sequence_type}")

        rnn_cls: type[nn.LSTM] | type[nn.GRU]
        if rnn_type == "lstm":
            rnn_cls = nn.LSTM
        elif rnn_type == "gru":
            rnn_cls = nn.GRU
        else:
            raise ValueError(f"Unknown rnn_type: {rnn_type}")

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.rnn = rnn_cls(
            input_size=self.tokenizer.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        direction_factor = 2 if bidirectional else 1
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * direction_factor, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence = self.tokenizer(x)
        output, _state = self.rnn(sequence)
        last = output[:, -1, :]
        return self.head(last)


def build_image_sequence_model(
    experiment: str,
    hidden_dim: int = 128,
    num_layers: int = 1,
    patch_size: int = 7,
    dropout: float = 0.1,
    bidirectional: bool = False,
) -> ImageSequenceClassifier:
    configs = {
        "lstm_row": {"rnn_type": "lstm", "sequence_type": "row"},
        "gru_row": {"rnn_type": "gru", "sequence_type": "row"},
        "lstm_patch": {"rnn_type": "lstm", "sequence_type": "patch"},
        "gru_patch": {"rnn_type": "gru", "sequence_type": "patch"},
    }
    if experiment not in configs:
        raise ValueError(f"Unknown image sequence experiment: {experiment}")

    return ImageSequenceClassifier(
        **configs[experiment],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        patch_size=patch_size,
        dropout=dropout,
        bidirectional=bidirectional,
    )

