from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn

# -----------------------------------------------------------------------------
# ViT Building Blocks (Back-prop variant)
# -----------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """Split image into patches and embed them."""

    def __init__(
        self,
        img_size: int = 96,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,C,H,W)
        x = self.proj(x)           # (B,embed_dim,H/ps,W/ps)
        x = x.flatten(2)           # (B,embed_dim,N)
        x = x.transpose(1, 2)      # (B,N,embed_dim)
        return x


class MLP(nn.Module):
    """Feed-forward network (FFN) for Transformer blocks."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        p: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SelfAttention(nn.Module):
    """Multi-head self-attention layer."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x)  # (B, N, 3*C)
        qkv = (
            qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
                .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        p: float = 0.0,
        norm_layer=nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SelfAttention(dim, num_heads, dropout=p)
        self.norm2 = norm_layer(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, hidden_dim, dim, p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTBP(nn.Module):
    """Vision Transformer using standard back-propagation.

    All hyper-parameters default to the same values as ``ViTFA`` for
    drop-in replacement.
    """

    def __init__(
        self,
        img_size: int = 96,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 2,
        embed_dim: int = 64,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        p: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        n_patches = self.patch_embed.n_patches

        # Class token & positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, p)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,C,H,W)
        B = x.shape[0]
        x = self.patch_embed(x)               # (B,N,D)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B,1,D)
        x = torch.cat((cls_tokens, x), dim=1)          # (B,N+1,D)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        logits = self.head(x[:, 0])  # CLS token output
        return logits


__all__ = ["ViTBP"] 