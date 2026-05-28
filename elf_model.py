from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import MapEncoder, NUM_ACTION_TOKENS, PAD_TOKEN_ID, SinusoidalPositionEmbedding, SinusoidalTimeEmbedding


@dataclass
class ELFTrainingConfig:
    denoiser_p_mean: float = -1.5
    denoiser_p_std: float = 0.8
    denoiser_noise_scale: float = 2.0
    decoder_prob: float = 0.2
    decoder_noise_scale: float = 1.0
    decoder_p_mean: float = 0.8
    decoder_p_std: float = 0.8
    self_cond_prob: float = 0.5
    self_cond_cfg_min: float = 0.5
    self_cond_cfg_max: float = 5.0
    t_eps: float = 5e-2
    grad_clip_norm: float = 1.0


def sample_timesteps(
    batch_size: int,
    *,
    p_mean: float,
    p_std: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    z = torch.randn((batch_size,), dtype=dtype, device=device) * p_std + p_mean
    return torch.sigmoid(z)


def sample_cfg_scale(
    batch_size: int,
    *,
    cfg_min: float,
    cfg_max: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    u = torch.rand((batch_size,), dtype=dtype, device=device)
    lo = float(1.0 + cfg_min)
    hi = float(1.0 + cfg_max)
    return lo * torch.exp(u * torch.tensor(hi / lo, dtype=dtype, device=device).log()) - 1.0


def restore_cond(z: torch.Tensor, clean: torch.Tensor, cond_seq_mask: torch.Tensor) -> torch.Tensor:
    mask = cond_seq_mask
    while mask.dim() < z.dim():
        mask = mask.unsqueeze(-1)
    return torch.where(mask > 0, clean, z)


def add_noise(
    x0: torch.Tensor,
    noise: torch.Tensor,
    t: torch.Tensor,
    cfg: ELFTrainingConfig,
    *,
    cond_seq_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    t_view = t.reshape(-1, 1, 1)
    z = t_view * x0 + (1.0 - t_view) * noise * cfg.denoiser_noise_scale
    if cond_seq_mask is not None:
        z = restore_cond(z, x0, cond_seq_mask)
    return z


def net_out_to_v_x(net_out: torch.Tensor, z: torch.Tensor, t: torch.Tensor, t_eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    x = net_out
    denom = torch.clamp(1.0 - t.reshape(-1, 1, 1), min=t_eps)
    return (x - z) / denom, x


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


class ELFActionTransformer(nn.Module):
    """ELF-style denoiser/decoder for grid action sequences.

    This mirrors the core ELF training surface in a small action-token setting:
    denoising predicts clean action embeddings, decoder mode predicts action
    logits, and self-conditioning can feed a previous clean prediction back in.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        ff_dim: int = 128,
        vocab_size: int = NUM_ACTION_TOKENS,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.action_embed = nn.Embedding(vocab_size, embed_dim)
        self.map_encoder = MapEncoder(out_dim=embed_dim)
        self.pos_embed = SinusoidalPositionEmbedding(embed_dim)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.self_cond_cfg_embed = nn.Sequential(
            SinusoidalTimeEmbedding(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.self_cond_proj = nn.Linear(2 * embed_dim, embed_dim)
        self.decoder_mode_embed = nn.Parameter(torch.zeros(embed_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.denoise_head = nn.Linear(embed_dim, embed_dim)
        self.decoder_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, vocab_size),
        )

    def embed_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.dtype != torch.long:
            actions = actions.long()
        return self.action_embed(actions)

    def action_logits_from_embeddings(self, seq_emb: torch.Tensor) -> torch.Tensor:
        return seq_emb @ self.action_embed.weight.transpose(0, 1)

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        map_tensor: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        *,
        self_cond_cfg_scale: Optional[torch.Tensor] = None,
        decoder_step_active: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, length, _ = z.shape
        if z.shape[-1] == 2 * self.embed_dim:
            z = self.self_cond_proj(z)

        pos_emb = self.pos_embed(length, z.device, z.dtype).unsqueeze(0).expand(batch, length, -1)
        t_emb = self.time_embed(t).unsqueeze(1).expand(batch, length, -1)
        map_emb = self.map_encoder(map_tensor).unsqueeze(1).expand(batch, length, -1)
        hidden = z + pos_emb + t_emb + map_emb

        if self_cond_cfg_scale is not None:
            cfg_emb = self.self_cond_cfg_embed(self_cond_cfg_scale).unsqueeze(1).expand(batch, length, -1)
            hidden = hidden + cfg_emb

        if decoder_step_active is not None:
            gate = decoder_step_active.to(hidden.dtype).reshape(-1, 1, 1)
            hidden = hidden + gate * self.decoder_mode_embed.reshape(1, 1, -1)

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = mask < 0.5

        if deterministic:
            was_training = self.transformer.training
            self.transformer.eval()
        try:
            hidden = self.transformer(hidden, src_key_padding_mask=key_padding_mask)
        finally:
            if deterministic and was_training:
                self.transformer.train()

        x_pred = self.denoise_head(hidden)
        decoder_logits = self.decoder_head(hidden)
        return x_pred, decoder_logits


def elf_action_loss(
    model: ELFActionTransformer,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    cfg: ELFTrainingConfig | None = None,
    *,
    generator: Optional[torch.Generator] = None,
    return_debug: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, object]]:
    """Compute the mixed ELF objective for one mini-batch.

    The batch may optionally contain `cond_seq_mask` for clean prefix positions.
    Without it, all non-PAD action tokens are denoising/decoder targets.
    """

    cfg = cfg or ELFTrainingConfig()
    map_tensor = batch["map"].to(device)
    clean = batch["clean_actions"].to(device).long()
    valid_mask = (clean != PAD_TOKEN_ID).float()
    cond_seq_mask = batch.get("cond_seq_mask")
    if cond_seq_mask is None:
        cond_seq_mask = torch.zeros_like(valid_mask)
    cond_seq_mask = cond_seq_mask.to(device=device, dtype=torch.float32)
    loss_mask = valid_mask * (1.0 - cond_seq_mask)
    cond_seq_mask_b = cond_seq_mask.unsqueeze(-1)

    x0 = model.embed_actions(clean)
    dtype = x0.dtype
    batch_size, seq_length = clean.shape

    t = sample_timesteps(
        batch_size,
        p_mean=cfg.denoiser_p_mean,
        p_std=cfg.denoiser_p_std,
        device=device,
        dtype=dtype,
    )
    noise = torch.randn_like(x0)
    denoiser_z = add_noise(x0, noise, t, cfg, cond_seq_mask=cond_seq_mask_b)

    decoder_step_active = torch.bernoulli(
        torch.full((batch_size,), cfg.decoder_prob, dtype=torch.float32),
        generator=generator,
    ).to(device=device, dtype=dtype)
    decoder_mask_b11 = decoder_step_active.reshape(-1, 1, 1)
    decoder_mask_b1 = decoder_step_active.reshape(-1, 1)

    decoder_lambda = torch.sigmoid(
        torch.randn((batch_size, seq_length, 1), dtype=dtype, device=device) * cfg.decoder_p_std
        + cfg.decoder_p_mean
    )
    decoder_noise = torch.randn_like(x0) * cfg.decoder_noise_scale
    decoder_z = decoder_lambda * x0 + (1.0 - decoder_lambda) * decoder_noise

    denoiser_t = t
    decoder_t = torch.ones_like(t)
    t_mixed = decoder_step_active * decoder_t + (1.0 - decoder_step_active) * denoiser_t
    z_mixed = decoder_mask_b11 * decoder_z + (1.0 - decoder_mask_b11) * denoiser_z

    v_target = (x0 - denoiser_z) / torch.clamp(1.0 - denoiser_t.reshape(-1, 1, 1), min=cfg.t_eps)
    use_self_cond = cfg.self_cond_prob > 0.0
    use_self_cond_mask = None
    if use_self_cond:
        use_self_cond_mask = (
            torch.rand((batch_size,), dtype=dtype, device=device) < cfg.self_cond_prob
        ).reshape(-1, 1, 1).to(dtype)

    self_cond_cfg_scale = sample_cfg_scale(
        batch_size,
        cfg_min=cfg.self_cond_cfg_min,
        cfg_max=cfg.self_cond_cfg_max,
        device=device,
        dtype=dtype,
    )

    def run_deterministic(z_in: torch.Tensor, t_in: torch.Tensor, self_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        model_in = z_in if self_cond is None else torch.cat([z_in, self_cond], dim=-1)
        with torch.no_grad():
            x_pred, _ = model(
                model_in,
                t_in,
                map_tensor,
                valid_mask,
                self_cond_cfg_scale=self_cond_cfg_scale,
                deterministic=True,
            )
        return x_pred

    shared_x_uncond = None
    if use_self_cond:
        z_uncond = restore_cond(torch.zeros_like(denoiser_z), x0, cond_seq_mask_b)
        shared_x_uncond = run_deterministic(denoiser_z, denoiser_t, z_uncond)
        x_pred_init = restore_cond(shared_x_uncond, x0, cond_seq_mask_b)
        x_self_cond = restore_cond(x_pred_init * use_self_cond_mask, x0, cond_seq_mask_b)
        x_self_cond = x_self_cond * (1.0 - decoder_mask_b11)
        model_input = torch.cat([z_mixed, x_self_cond], dim=-1)
    else:
        model_input = z_mixed

    x_pred, decoder_logits = model(
        model_input,
        t_mixed,
        map_tensor,
        valid_mask,
        self_cond_cfg_scale=self_cond_cfg_scale,
        decoder_step_active=decoder_step_active,
        deterministic=False,
    )

    log_probs = F.log_softmax(decoder_logits.float(), dim=-1)
    ce_per_token = -log_probs.gather(-1, clean.unsqueeze(-1)).squeeze(-1)

    v_pred, _ = net_out_to_v_x(x_pred, denoiser_z, denoiser_t, cfg.t_eps)
    v_final_target = v_target
    if use_self_cond and shared_x_uncond is not None:
        x_cond = run_deterministic(denoiser_z, denoiser_t, restore_cond(shared_x_uncond, x0, cond_seq_mask_b))
        v_cond, _ = net_out_to_v_x(x_cond, denoiser_z, denoiser_t, cfg.t_eps)
        v_uncond, _ = net_out_to_v_x(shared_x_uncond, denoiser_z, denoiser_t, cfg.t_eps)
        sc_w = self_cond_cfg_scale.reshape(batch_size, 1, 1)
        sc_guidance = (1.0 - 1.0 / sc_w) * (v_cond - v_uncond)
        sc_guidance = torch.where(use_self_cond_mask.bool(), sc_guidance, torch.zeros_like(sc_guidance))
        v_final_target = (v_target + sc_guidance).detach()

    l2_per_token = ((v_pred - v_final_target) ** 2).mean(dim=-1)
    ce_mask = loss_mask * decoder_mask_b1
    l2_mask = loss_mask * (1.0 - decoder_mask_b1)

    total = (ce_per_token * ce_mask).sum() + (l2_per_token * l2_mask).sum()
    loss = total / loss_mask.sum().clamp_min(1.0)

    if not return_debug:
        return loss

    with torch.no_grad():
        decoded_logits = model.action_logits_from_embeddings(x_pred[0])
        debug = {
            "loss": float(loss.detach().cpu()),
            "ce_loss": float(masked_mean(ce_per_token.detach(), ce_mask).cpu()),
            "l2_loss": float(masked_mean(l2_per_token.detach(), l2_mask).cpu()),
            "decoder_fraction": float(decoder_step_active.float().mean().detach().cpu()),
            "t0": float(t[0].detach().cpu()),
            "clean": clean[0].detach().cpu(),
            "pred_tokens": decoded_logits.argmax(dim=-1).detach().cpu(),
            "mse_head": l2_per_token[0, :10].detach().cpu(),
        }
    return loss, debug


def elf_optimizer_step(
    model: ELFActionTransformer,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    cfg: ELFTrainingConfig | None = None,
    *,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    loss, debug = elf_action_loss(model, batch, device, cfg, generator=generator, return_debug=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=(cfg or ELFTrainingConfig()).grad_clip_norm)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return {
        "loss": float(loss.detach().cpu()),
        "ce_loss": float(debug["ce_loss"]),
        "l2_loss": float(debug["l2_loss"]),
        "decoder_fraction": float(debug["decoder_fraction"]),
    }
