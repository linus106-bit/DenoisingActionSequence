from __future__ import annotations

import torch
import torch.nn as nn

from .common import MapEncoder, SinusoidalPositionEmbedding, SinusoidalTimeEmbedding


class FlowMatchingTransformer(nn.Module):
    def __init__(self, embed_dim: int = 64, n_heads: int = 4, n_layers: int = 3, ff_dim: int = 128, max_actions: int = 7):
        super().__init__()
        self.embed_dim = embed_dim
        self.action_embed = nn.Embedding(max_actions, embed_dim)
        self.map_encoder = MapEncoder(out_dim=embed_dim)
        self.pos_embed = SinusoidalPositionEmbedding(embed_dim)
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
        if actions.dtype != torch.long:
            actions = actions.long()
        return self.action_embed(actions)

    def action_logits_from_embeddings(self, seq_emb: torch.Tensor) -> torch.Tensor:
        action_table = self.action_embed.weight
        return seq_emb @ action_table.transpose(0, 1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, map_tensor: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, length, _ = x_t.shape
        pos_emb = self.pos_embed(length, x_t.device, x_t.dtype).unsqueeze(0).expand(batch, length, -1)
        t_emb = self.time_embed(t).unsqueeze(1).expand(batch, length, -1)
        m_emb = self.map_encoder(map_tensor).unsqueeze(1).expand(batch, length, -1)
        hidden = x_t + pos_emb + t_emb + m_emb
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = mask < 0.5
        hidden = self.transformer(hidden, src_key_padding_mask=key_padding_mask)
        return self.out(hidden)
