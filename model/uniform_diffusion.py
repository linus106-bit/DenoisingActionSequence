from __future__ import annotations

import torch

from .common import EOS_TOKEN_ID, NUM_ACTION_TOKENS
from .masked_diffusion import MaskedDiffusionTrajectoryTransformer


def uniform_forward_process(
    input_ids: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    min_noise_prob: float = 1e-3,
    max_noise_prob: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply uniform-token corruption to a batch of token ids.

    Each sequence samples one corruption probability uniformly from
    ``[min_noise_prob, max_noise_prob]`` and corrupts valid tokens
    independently with that probability. Corrupted positions are replaced by a
    uniformly sampled *different* action/EOS token instead of a dedicated mask
    token. ``valid_mask`` can be used to keep PAD tokens uncorrupted and
    unsupervised.
    """
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be a 2D tensor, got shape {tuple(input_ids.shape)}")
    if not 0.0 <= min_noise_prob <= max_noise_prob <= 1.0:
        raise ValueError(
            "noise probabilities must satisfy 0.0 <= min_noise_prob <= max_noise_prob <= 1.0, "
            f"got min_noise_prob={min_noise_prob} and max_noise_prob={max_noise_prob}"
        )

    batch, length = input_ids.shape
    t = torch.rand(batch, device=input_ids.device)
    p_noise = (max_noise_prob - min_noise_prob) * t + min_noise_prob
    p_noise = p_noise[:, None].expand(batch, length)

    noisy_indices = torch.rand((batch, length), device=input_ids.device) < p_noise
    if valid_mask is not None:
        if valid_mask.shape != input_ids.shape:
            raise ValueError(
                f"valid_mask must have shape {tuple(input_ids.shape)}, got {tuple(valid_mask.shape)}"
            )
        noisy_indices = noisy_indices & valid_mask.bool()

    noisy_batch = input_ids.clone()
    if bool(noisy_indices.any()):
        original = input_ids[noisy_indices]
        # Clean action sequences contain movement tokens 1..4 plus EOS=5 at
        # valid positions. Sample a non-zero delta modulo that set so the
        # replacement is guaranteed to differ from the original token.
        delta = torch.randint(1, EOS_TOKEN_ID, size=original.shape, device=input_ids.device)
        noisy_batch[noisy_indices] = ((original - 1 + delta) % EOS_TOKEN_ID) + 1
    return noisy_batch, noisy_indices, p_noise


class UniformDiffusionTrajectoryTransformer(MaskedDiffusionTrajectoryTransformer):
    """Map-conditioned uniform-token diffusion transformer for actions.

    This model shares the masked diffusion architecture and objective, but its
    inputs are corrupted with random action/EOS tokens rather than a dedicated
    mask token. Because the mask token is never used, the embedding and output
    vocabulary covers only the base action vocabulary ``0..6``.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        ff_dim: int = 128,
        vocab_size: int = NUM_ACTION_TOKENS,
    ):
        super().__init__(
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_dim=ff_dim,
            vocab_size=vocab_size,
        )
        self.mask_token_id = None
