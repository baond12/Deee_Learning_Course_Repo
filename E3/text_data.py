from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


TOY_SENTIMENT_SAMPLES: list[tuple[str, int]] = [
    ("this movie is excellent and inspiring", 1),
    ("the product works great and feels reliable", 1),
    ("i love this course and the lectures are clear", 1),
    ("the model converged fast and the result is good", 1),
    ("amazing service friendly staff and clean room", 1),
    ("the food was delicious and worth the price", 1),
    ("bai hoc nay rat hay va de hieu", 1),
    ("ket qua tot hon mong doi", 1),
    ("this movie is boring and too long", 0),
    ("the product broke after one day", 0),
    ("i hate the interface because it is confusing", 0),
    ("the model failed and the accuracy is poor", 0),
    ("terrible service rude staff and dirty room", 0),
    ("the food was cold and disappointing", 0),
    ("bai hoc nay kho hieu va qua dai", 0),
    ("ket qua te hon mong doi", 0),
]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


@dataclass(frozen=True)
class Vocabulary:
    stoi: dict[str, int]
    itos: list[str]

    @property
    def pad_index(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def unk_index(self) -> int:
        return self.stoi[UNK_TOKEN]

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(token, self.unk_index) for token in tokenize(text)]


def build_vocab(samples: list[tuple[str, int]], min_freq: int = 1) -> Vocabulary:
    counts: dict[str, int] = {}
    for text, _label in samples:
        for token in tokenize(text):
            counts[token] = counts.get(token, 0) + 1

    itos = [PAD_TOKEN, UNK_TOKEN]
    itos.extend(sorted(token for token, count in counts.items() if count >= min_freq))
    stoi = {token: index for index, token in enumerate(itos)}
    return Vocabulary(stoi=stoi, itos=itos)


class SentimentDataset(Dataset):
    def __init__(self, samples: list[tuple[str, int]], vocab: Vocabulary) -> None:
        self.samples = samples
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text, label = self.samples[index]
        encoded = self.vocab.encode(text)
        if not encoded:
            encoded = [self.vocab.unk_index]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(len(encoded), dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )


def collate_sentiment_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    pad_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens, lengths, labels = zip(*batch)
    padded = pad_sequence(tokens, batch_first=True, padding_value=pad_index)
    return padded, torch.stack(lengths), torch.stack(labels)

