from __future__ import annotations

import torch
import torch.nn as nn

from .common import (
    MASK_TOKEN_ID,
    VOCAB_SIZE_WITH_MASK,
    MapEncoder,
    SinusoidalPositionEmbedding,
    SinusoidalTimeEmbedding,
)


class MaskedDiffusionTrajectoryTransformer(nn.Module):
    """Map-conditioned masked-token denoising transformer for action sequences.

    The model receives an action sequence where some non-PAD tokens have been
    replaced by ``MASK_TOKEN_ID``. A scalar timestep / mask ratio ``t`` conditions
    how aggressively the sequence was corrupted. The output is per-position
    logits over the action vocabulary, including the mask token; training losses
    can ignore PAD and unmasked positions as needed.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        ff_dim: int = 128,
        vocab_size: int = VOCAB_SIZE_WITH_MASK,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.mask_token_id = MASK_TOKEN_ID
        self.action_embed = nn.Embedding(vocab_size, embed_dim)
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
        self.out = nn.Linear(embed_dim, vocab_size)

    def forward(
        self,
        masked_actions: torch.Tensor,
        t: torch.Tensor,
        map_tensor: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if masked_actions.dtype != torch.long:
            masked_actions = masked_actions.long()
        if t.dim() == 0:
            t = t.expand(masked_actions.shape[0])
        else:
            t = t.reshape(-1)
        batch, length = masked_actions.shape
        token_emb = self.action_embed(masked_actions)
        pos_emb = self.pos_embed(length, token_emb.device, token_emb.dtype).unsqueeze(0).expand(batch, length, -1)
        time_emb = (
            self.time_embed(t.to(device=token_emb.device, dtype=token_emb.dtype))
            .unsqueeze(1)
            .expand(batch, length, -1)
        )
        map_emb = self.map_encoder(map_tensor).unsqueeze(1).expand(batch, length, -1)
        hidden = token_emb + pos_emb + time_emb + map_emb
        hidden = self.transformer(hidden, src_key_padding_mask=pad_mask)
        return self.out(hidden)
