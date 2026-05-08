from __future__ import annotations

import torch
import torch.nn as nn

from .common import (
    MASK_TOKEN_ID,
    VOCAB_SIZE_WITH_MASK,
    MapEncoder,
    SinusoidalPositionEmbedding,
)


def forward_process(
    input_ids: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    min_mask_prob: float = 1e-3,
    max_mask_prob: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply LLaDA-style random masking to a batch of token ids.

    Each sequence samples one masking probability uniformly from
    ``[min_mask_prob, max_mask_prob]`` and masks tokens independently with that
    probability. ``valid_mask`` can be used to keep PAD tokens uncorrupted and
    unsupervised.
    """
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be a 2D tensor, got shape {tuple(input_ids.shape)}")
    if not 0.0 <= min_mask_prob <= max_mask_prob <= 1.0:
        raise ValueError(
            "mask probabilities must satisfy 0.0 <= min_mask_prob <= max_mask_prob <= 1.0, "
            f"got min_mask_prob={min_mask_prob} and max_mask_prob={max_mask_prob}"
        )

    batch, length = input_ids.shape
    t = torch.rand(batch, device=input_ids.device)
    p_mask = (max_mask_prob - min_mask_prob) * t + min_mask_prob
    p_mask = p_mask[:, None].expand(batch, length)

    masked_indices = torch.rand((batch, length), device=input_ids.device) < p_mask
    if valid_mask is not None:
        if valid_mask.shape != input_ids.shape:
            raise ValueError(
                f"valid_mask must have shape {tuple(input_ids.shape)}, got {tuple(valid_mask.shape)}"
            )
        masked_indices = masked_indices & valid_mask.bool()

    noisy_batch = torch.where(masked_indices, torch.full_like(input_ids, MASK_TOKEN_ID), input_ids)
    return noisy_batch, masked_indices, p_mask


class MaskedDiffusionTrajectoryTransformer(nn.Module):
    """Map-conditioned LLaDA-style masked-token transformer for actions.

    The model receives an action sequence where selected non-PAD tokens have
    been replaced by ``MASK_TOKEN_ID``. Like LLaDA, the backbone is a
    bidirectional Transformer encoder without causal attention and without any
    timestep embedding. Training code supervises the masked positions and can
    reweight losses by their sampled mask probabilities.
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
        map_tensor: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if masked_actions.dtype != torch.long:
            masked_actions = masked_actions.long()
        batch, length = masked_actions.shape
        token_emb = self.action_embed(masked_actions)
        pos_emb = self.pos_embed(length, token_emb.device, token_emb.dtype).unsqueeze(0).expand(batch, length, -1)
        map_emb = self.map_encoder(map_tensor).unsqueeze(1).expand(batch, length, -1)
        hidden = token_emb + pos_emb + map_emb
        hidden = self.transformer(hidden, src_key_padding_mask=pad_mask)
        return self.out(hidden)
