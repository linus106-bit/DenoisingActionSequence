from __future__ import annotations

import torch
import torch.nn as nn

from .common import BOS_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID, MapEncoder, SinusoidalPositionEmbedding


class AutoregressiveTrajectoryTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        ff_dim: int = 128,
        max_actions: int = 7,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.action_embed = nn.Embedding(max_actions, embed_dim)
        self.map_encoder = MapEncoder(out_dim=embed_dim)
        self.pos_embed = SinusoidalPositionEmbedding(embed_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out = nn.Linear(embed_dim, max_actions)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((length, length), float("-inf"), device=device), diagonal=1)

    def forward(self, tokens: torch.Tensor, map_tensor: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        if tokens.dtype != torch.long:
            tokens = tokens.long()
        batch, length = tokens.shape
        token_emb = self.action_embed(tokens)
        pos_emb = self.pos_embed(length, token_emb.device, token_emb.dtype).unsqueeze(0).expand(batch, length, -1)
        map_emb = self.map_encoder(map_tensor).unsqueeze(1).expand(batch, length, -1)
        hidden = token_emb + pos_emb + map_emb
        hidden = self.transformer(
            hidden,
            mask=self._causal_mask(length, tokens.device),
            src_key_padding_mask=pad_mask,
        )
        return self.out(hidden)

    @torch.no_grad()
    def generate(
        self,
        map_tensor: torch.Tensor,
        max_len: int,
        decode: str = "argmax",
        temperature: float = 1.0,
    ) -> torch.Tensor:
        device = map_tensor.device
        batch = map_tensor.shape[0]
        tokens = torch.full((batch, 1), BOS_TOKEN_ID, dtype=torch.long, device=device)
        finished = torch.zeros((batch,), dtype=torch.bool, device=device)

        for _ in range(max_len):
            logits = self(tokens, map_tensor)[:, -1, :]
            logits[:, BOS_TOKEN_ID] = float("-inf")
            if temperature <= 0:
                raise ValueError("temperature must be positive")
            if decode == "sample":
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            elif decode == "argmax":
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                raise ValueError(f"Unsupported decode mode: {decode}")
            next_token[finished] = EOS_TOKEN_ID
            tokens = torch.cat([tokens, next_token], dim=1)
            finished = finished | (next_token.squeeze(1) == EOS_TOKEN_ID)
            if bool(finished.all()):
                break

        generated = tokens[:, 1:]
        if generated.shape[1] < max_len:
            pad = torch.full(
                (batch, max_len - generated.shape[1]),
                PAD_TOKEN_ID,
                dtype=torch.long,
                device=device,
            )
            generated = torch.cat([generated, pad], dim=1)
        return generated[:, :max_len]
