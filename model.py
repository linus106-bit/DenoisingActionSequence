from __future__ import annotations

import math

import torch
import torch.nn as nn

EOS_TOKEN_ID = 5
PAD_ACTION_VALUE = 6
PAD_TOKEN_ID = 6


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,)
        half = self.dim // 2
        freq = torch.exp(
            torch.arange(half, device=t.device, dtype=t.dtype) * (-math.log(10000.0) / max(half - 1, 1))
        )
        args = t[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class MapEncoder(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FlowMatchingTransformer(nn.Module):
    def __init__(self, embed_dim: int = 64, n_heads: int = 4, n_layers: int = 3, ff_dim: int = 128, max_actions: int = 7):
        super().__init__()
        self.embed_dim = embed_dim
        # token ids: 0 unused, 1..4 real actions, 5=EOS, 6=PAD
        self.action_embed = nn.Embedding(max_actions, embed_dim)
        self.map_encoder = MapEncoder(out_dim=embed_dim)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out = nn.Linear(embed_dim, embed_dim)

    def embed_actions(self, actions: torch.Tensor) -> torch.Tensor:
        # actions already use 1..4 + EOS(5) + PAD(6)
        if actions.dtype != torch.long:
            actions = actions.long()
        return self.action_embed(actions)

    def action_logits_from_embeddings(self, seq_emb: torch.Tensor) -> torch.Tensor:
        """
        Convert sequence embeddings to token logits (0~6, EOS/PAD 포함) using
        the action embedding table as a tied output projection.
        """
        action_table = self.action_embed.weight  # (7, D): 0 unused, 1..4 + EOS(5) + PAD(6)
        return seq_emb @ action_table.transpose(0, 1)  # (B, L, 7) or (L, 7)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, map_tensor: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x_t: (B, L, D), t: (B,), map_tensor: (B,3,10,10)
        B, L, _ = x_t.shape
        t_emb = self.time_embed(t).unsqueeze(1).expand(B, L, -1)
        m_emb = self.map_encoder(map_tensor).unsqueeze(1).expand(B, L, -1)

        h = x_t + t_emb + m_emb
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = mask < 0.5
        h = self.transformer(h, src_key_padding_mask=key_padding_mask)
        return self.out(h)
