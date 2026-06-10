from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from data_utils import EOS_ACTION, GridDenoiseDataset, PAD_ACTION
from elf_model import ELFActionTransformer, ELFTrainingConfig, elf_action_loss
from model import (
    MASK_TOKEN_ID,
    AutoregressiveTrajectoryTransformer,
    BOS_TOKEN_ID,
    FlowMatchingTransformer,
    MaskedDiffusionTrajectoryTransformer,
    build_warmup_cosine_lr_scheduler,
    forward_process,
)


def _tokenize_for_print(tokens: torch.Tensor) -> list[int]:
    return [int(t) for t in tokens.detach().cpu().tolist()]


def _make_t_scaled_noisy(
    clean: torch.Tensor,
    valid_mask: torch.Tensor,
    t: torch.Tensor,
    pad_noise_prob: float = 1.0,
) -> torch.Tensor:
    """
    Build x0 tokens from clean actions.
    - Valid positions: replace exactly floor(valid_len * (1 - t)) positions with
      a different token in {1,2,3,4,EOS}.
    This keeps the FM convention aligned with evaluation:
    t=0 is most noisy, t=1 is clean.
    PAD positions stay as PAD and are excluded from loss, while EOS is supervised.
    """
    del pad_noise_prob  # PAD is no longer supervised or corrupted during training.
    noisy = clean.clone()
    batch = clean.shape[0]
    for i in range(batch):
        noise_level = 1.0 - float(t[i].item())
        valid_idx = torch.nonzero(valid_mask[i] > 0.5, as_tuple=False).squeeze(-1)
        valid_len = int(valid_idx.numel())
        n_replace = int(valid_len * noise_level)
        if valid_len == 0 or n_replace <= 0:
            pass
        else:
            perm = torch.randperm(valid_len, device=clean.device)
            chosen = valid_idx[perm[:n_replace]]
            original = clean[i, chosen]
            # Ensure replacement token differs from the original token (1~4 or EOS).
            delta = torch.randint(1, EOS_ACTION, size=original.shape, device=clean.device)
            noisy[i, chosen] = ((original - 1 + delta) % EOS_ACTION) + 1
    return noisy


def fm_loss(model, batch, device, return_debug: bool = False, pad_noise_prob: float = 1.0):
    map_tensor = batch["map"].to(device)
    clean = batch["clean_actions"].to(device)
    valid_mask = (clean != PAD_ACTION).float()
    mask = valid_mask

    t = torch.rand(clean.shape[0], device=device)
    noisy = _make_t_scaled_noisy(clean, valid_mask, t, pad_noise_prob=pad_noise_prob)

    x0 = model.embed_actions(noisy)
    x1 = model.embed_actions(clean)

    xt = (1.0 - t[:, None, None]) * x0 + t[:, None, None] * x1
    u_t = x1 - x0

    pred_v = model(xt, t, map_tensor, mask)
    mse = (pred_v - u_t).pow(2).mean(dim=-1)
    loss = (mse * mask).sum() / (mask.sum() + 1e-6)
    if not return_debug:
        return loss

    pred_next = x0 + pred_v
    pred_logits = model.action_logits_from_embeddings(pred_next[0])
    pred_tokens = pred_logits.argmax(dim=-1)
    debug = {
        "noisy": noisy[0].detach().cpu(),
        "clean": clean[0].detach().cpu(),
        "pred_tokens": pred_tokens.detach().cpu(),
        "t0": float(t[0].item()),
        "mse_head": mse[0, :10].detach().cpu(),
        "loss": float(loss.item()),
    }
    return loss, debug


def ar_loss(model: AutoregressiveTrajectoryTransformer, batch: dict, device: torch.device) -> torch.Tensor:
    map_tensor = batch["map"].to(device)
    clean = batch["clean_actions"].to(device)
    bos = torch.full((clean.shape[0], 1), BOS_TOKEN_ID, dtype=torch.long, device=device)
    tokens_in = torch.cat([bos, clean[:, :-1]], dim=1)
    logits = model(tokens_in, map_tensor)
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        clean.reshape(-1),
        ignore_index=PAD_ACTION,
    )


def masked_diffusion_loss(
    model: MaskedDiffusionTrajectoryTransformer,
    batch: dict,
    device: torch.device,
    min_mask_prob: float = 0.15,
    max_mask_prob: float = 1.0,
) -> torch.Tensor:
    map_tensor = batch["map"].to(device)
    clean = batch["clean_actions"].to(device)
    valid_mask = clean != PAD_ACTION
    masked, mask_positions, p_mask = forward_process(
        clean,
        valid_mask=valid_mask,
        min_mask_prob=min_mask_prob,
        max_mask_prob=max_mask_prob,
    )

    # Ensure each sequence contributes at least one supervised denoising target.
    for i in range(clean.shape[0]):
        if not bool(mask_positions[i].any()):
            candidates = torch.nonzero(valid_mask[i], as_tuple=False).squeeze(-1)
            if int(candidates.numel()) > 0:
                chosen = candidates[torch.randint(candidates.numel(), (1,), device=device)]
                mask_positions[i, chosen] = True
                masked[i, chosen] = MASK_TOKEN_ID

    pad_mask = clean == PAD_ACTION
    logits = model(masked, map_tensor, pad_mask=pad_mask)
    token_loss = F.cross_entropy(logits[mask_positions], clean[mask_positions], reduction="none")
    token_loss = token_loss / p_mask[mask_positions].clamp_min(1e-6)
    return token_loss.sum() / valid_mask.sum().clamp_min(1)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = GridDenoiseDataset(
        n_samples=args.n_samples,
        max_seq_len=args.max_seq_len,
        grid_size=args.grid_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    if args.model_type == "flow_matching":
        model = FlowMatchingTransformer(
            embed_dim=args.embed_dim,
            n_layers=args.layers,
            n_heads=args.heads,
            ff_dim=args.ff_dim,
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.model_type == "autoregressive":
        model = AutoregressiveTrajectoryTransformer(
            embed_dim=args.embed_dim,
            n_layers=args.layers,
            n_heads=args.heads,
            ff_dim=args.ff_dim,
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.model_type == "masked_diffusion":
        model = MaskedDiffusionTrajectoryTransformer(
            embed_dim=args.embed_dim,
            n_layers=args.layers,
            n_heads=args.heads,
            ff_dim=args.ff_dim,
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.model_type == "elf":
        model = ELFActionTransformer(
            embed_dim=args.embed_dim,
            n_layers=args.layers,
            n_heads=args.heads,
            ff_dim=args.ff_dim,
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elf_cfg = ELFTrainingConfig(
            denoiser_p_mean=args.elf_denoiser_p_mean,
            denoiser_p_std=args.elf_denoiser_p_std,
            denoiser_noise_scale=args.elf_denoiser_noise_scale,
            decoder_prob=args.elf_decoder_prob,
            decoder_noise_scale=args.elf_decoder_noise_scale,
            decoder_p_mean=args.elf_decoder_p_mean,
            decoder_p_std=args.elf_decoder_p_std,
            self_cond_prob=args.elf_self_cond_prob,
            self_cond_cfg_min=args.elf_self_cond_cfg_min,
            self_cond_cfg_max=args.elf_self_cond_cfg_max,
            t_eps=args.elf_t_eps,
            grad_clip_norm=args.elf_grad_clip_norm,
        )
    else:
        raise ValueError(f"Unsupported --model_type: {args.model_type}")

    total_steps = max(args.epochs * len(loader), 1)
    scheduler, warmup_steps = build_warmup_cosine_lr_scheduler(
        opt,
        total_steps=total_steps,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
    )
    print(
        f"[LR] schedule=linear_warmup_cosine "
        f"total_steps={total_steps} warmup_steps={warmup_steps} "
        f"min_lr_ratio={args.min_lr_ratio}"
    )

    if args.model_type != "elf":
        elf_cfg = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            need_debug = args.model_type in ("flow_matching", "elf") and epoch == 1 and running == 0.0
            if args.model_type == "flow_matching" and need_debug:
                loss, dbg = fm_loss(model, batch, device, return_debug=True, pad_noise_prob=args.pad_noise_prob)
            elif args.model_type == "flow_matching":
                loss = fm_loss(model, batch, device, pad_noise_prob=args.pad_noise_prob)
            elif args.model_type == "autoregressive":
                loss = ar_loss(model, batch, device)
            elif args.model_type == "masked_diffusion":
                loss = masked_diffusion_loss(
                    model,
                    batch,
                    device,
                    min_mask_prob=args.mask_min_prob,
                    max_mask_prob=args.mask_max_prob,
                )
            elif args.model_type == "elf" and need_debug:
                loss, dbg = elf_action_loss(model, batch, device, cfg=elf_cfg, return_debug=True)
            else:
                loss = elf_action_loss(model, batch, device, cfg=elf_cfg)
            loss.backward()
            if args.model_type in ("autoregressive", "masked_diffusion"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            elif args.model_type == "elf" and elf_cfg is not None and elf_cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=elf_cfg.grad_clip_norm)
            opt.step()
            if scheduler is not None:
                scheduler.step()
            running += loss.item()

            if need_debug:
                print("[Debug:first-step] t:", round(dbg["t0"], 4))
                if args.model_type == "flow_matching":
                    print("[Debug:first-step] noisy[0]:", dbg["noisy"].tolist())
                elif args.model_type == "elf":
                    print("[Debug:first-step] elf_ce_loss:", round(dbg["ce_loss"], 6))
                    print("[Debug:first-step] elf_l2_loss:", round(dbg["l2_loss"], 6))
                    print("[Debug:first-step] elf_decoder_fraction:", round(dbg["decoder_fraction"], 6))
                print("[Debug:first-step] clean[0]:", dbg["clean"].tolist())
                print("[Debug:first-step] pred_token(argmax)[0]:", _tokenize_for_print(dbg["pred_tokens"]))
                print("[Debug:first-step] token_mse_head(first 10):", dbg["mse_head"].tolist())
                print("[Debug:first-step] loss:", round(dbg["loss"], 6))

        avg = running / max(len(loader), 1)
        print(f"Epoch {epoch}/{args.epochs} - loss: {avg:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": vars(args)}, out)
    print(f"Saved checkpoint: {out}")


def apply_model_size_preset(args: argparse.Namespace) -> argparse.Namespace:
    config_path = Path(args.model_config)
    if not config_path.exists():
        raise FileNotFoundError(f"Model config file not found: {config_path}")

    config = yaml.safe_load(config_path.read_text())
    presets = config.get("model_sizes", {})
    if args.model_size not in presets:
        available = ", ".join(sorted(presets.keys()))
        raise ValueError(f"Unknown --model_size '{args.model_size}'. Available: {available}")

    preset = presets[args.model_size]
    for key in ("embed_dim", "ff_dim", "layers", "heads"):
        if key not in preset:
            raise ValueError(f"Missing '{key}' in model size preset '{args.model_size}'")

    args.embed_dim = int(preset["embed_dim"])
    args.ff_dim = int(preset["ff_dim"])
    args.layers = int(preset["layers"])
    args.heads = int(preset["heads"])
    print(
        f"[ModelSize] preset={args.model_size} "
        f"embed_dim={args.embed_dim} ff_dim={args.ff_dim} "
        f"layers={args.layers} heads={args.heads}"
    )
    return args


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", type=str, choices=["flow_matching", "autoregressive", "masked_diffusion", "elf"], default="flow_matching")
    p.add_argument("--n_samples", type=int, default=1500)
    p.add_argument("--grid_size", type=int, default=8)
    p.add_argument("--max_seq_len", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--embed_dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ff_dim", type=int, default=128)
    p.add_argument("--model_config", type=str, default="config.yaml")
    p.add_argument("--model_size", type=str, default="base")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=-1)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--pad_noise_prob", type=float, default=1.0)
    p.add_argument("--mask_min_prob", type=float, default=0.15)
    p.add_argument("--mask_max_prob", type=float, default=1.0)
    p.add_argument("--elf_denoiser_p_mean", type=float, default=ELFTrainingConfig.denoiser_p_mean)
    p.add_argument("--elf_denoiser_p_std", type=float, default=ELFTrainingConfig.denoiser_p_std)
    p.add_argument("--elf_denoiser_noise_scale", type=float, default=ELFTrainingConfig.denoiser_noise_scale)
    p.add_argument("--elf_decoder_prob", type=float, default=ELFTrainingConfig.decoder_prob)
    p.add_argument("--elf_decoder_noise_scale", type=float, default=ELFTrainingConfig.decoder_noise_scale)
    p.add_argument("--elf_decoder_p_mean", type=float, default=ELFTrainingConfig.decoder_p_mean)
    p.add_argument("--elf_decoder_p_std", type=float, default=ELFTrainingConfig.decoder_p_std)
    p.add_argument("--elf_self_cond_prob", type=float, default=ELFTrainingConfig.self_cond_prob)
    p.add_argument("--elf_self_cond_cfg_min", type=float, default=ELFTrainingConfig.self_cond_cfg_min)
    p.add_argument("--elf_self_cond_cfg_max", type=float, default=ELFTrainingConfig.self_cond_cfg_max)
    p.add_argument("--elf_t_eps", type=float, default=ELFTrainingConfig.t_eps)
    p.add_argument("--elf_grad_clip_norm", type=float, default=ELFTrainingConfig.grad_clip_norm)
    p.add_argument("--out", type=str, default="checkpoints/fm_denoiser.pt")
    return p


if __name__ == "__main__":
    p = build_parser()
    args = p.parse_args()
    args = apply_model_size_preset(args)
    train(args)
