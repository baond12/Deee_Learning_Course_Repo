from __future__ import annotations

import math

import torch
from torch import nn


class PatchTokenizer(nn.Module):
    def __init__(self, patch_size: int = 7, embed_dim: int = 64) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_tokens = (28 // patch_size) * (28 // patch_size)
        self.projection = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.projection(x)
        return tokens.flatten(2).transpose(1, 2)


class RowTokenizer(nn.Module):
    def __init__(self, embed_dim: int = 64) -> None:
        super().__init__()
        self.num_tokens = 28
        self.projection = nn.Linear(28, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows = x.squeeze(1)
        return self.projection(rows)


class CustomMultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int = 64, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.out_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, embed_dim = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, token_count, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention = torch.softmax(scores, dim=-1)
        attention = self.attn_drop(attention)

        context = torch.matmul(attention, v)
        context = context.transpose(1, 2).contiguous().reshape(batch_size, token_count, embed_dim)
        return self.out_drop(self.out_proj(context))


class TorchMultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int = 64, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _weights = self.attention(x, x, x, need_weights=False)
        return output


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        ffn_dim: int = 128,
        attention_type: str = "custom",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if attention_type == "custom":
            attention = CustomMultiHeadSelfAttention(embed_dim, num_heads, dropout)
        elif attention_type == "pytorch":
            attention = TorchMultiHeadSelfAttention(embed_dim, num_heads, dropout)
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}")

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = attention
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop1(self.attention(self.norm1(x)))
        x = x + self.ffn(self.norm2(x))
        return x


class ImageTransformerClassifier(nn.Module):
    def __init__(
        self,
        tokenizer_type: str = "patch",
        attention_type: str = "custom",
        embed_dim: int = 64,
        num_heads: int = 4,
        ffn_dim: int = 128,
        depth: int = 1,
        patch_size: int = 7,
        num_classes: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if tokenizer_type == "patch":
            self.tokenizer = PatchTokenizer(patch_size=patch_size, embed_dim=embed_dim)
        elif tokenizer_type == "row":
            self.tokenizer = RowTokenizer(embed_dim=embed_dim)
        else:
            raise ValueError(f"Unknown tokenizer_type: {tokenizer_type}")

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.tokenizer.num_tokens + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    attention_type=attention_type,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    fan_in, _fan_out = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(module.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.pos_drop(tokens + self.pos_embedding)

        for block in self.blocks:
            tokens = block(tokens)

        cls_representation = tokens[:, 0]
        return self.head(cls_representation)


def build_model(
    experiment: str,
    embed_dim: int = 64,
    num_heads: int = 4,
    ffn_dim: int = 128,
    depth: int = 1,
    patch_size: int = 7,
    dropout: float = 0.1,
) -> ImageTransformerClassifier:
    configs = {
        "custom_patch": {"tokenizer_type": "patch", "attention_type": "custom"},
        "pytorch_patch": {"tokenizer_type": "patch", "attention_type": "pytorch"},
        "custom_row": {"tokenizer_type": "row", "attention_type": "custom"},
    }
    if experiment not in configs:
        raise ValueError(f"Unknown experiment: {experiment}")

    return ImageTransformerClassifier(
        **configs[experiment],
        embed_dim=embed_dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        depth=depth,
        patch_size=patch_size,
        dropout=dropout,
    )

