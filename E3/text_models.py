from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SentimentRNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_index: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1,
        rnn_type: str = "lstm",
        bidirectional: bool = True,
        dropout: float = 0.1,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_index)
        rnn_cls: type[nn.LSTM] | type[nn.GRU]
        if rnn_type == "lstm":
            rnn_cls = nn.LSTM
        elif rnn_type == "gru":
            rnn_cls = nn.GRU
        else:
            raise ValueError(f"Unknown rnn_type: {rnn_type}")

        self.rnn = rnn_cls(
            input_size=embed_dim,
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

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens)
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _state = self.rnn(packed)
        unpacked, _unpacked_lengths = pad_packed_sequence(packed_output, batch_first=True)

        max_len = unpacked.size(1)
        mask = torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
        masked = unpacked * mask.unsqueeze(-1)
        pooled = masked.sum(dim=1) / lengths.clamp_min(1).unsqueeze(1)
        return self.head(pooled)

