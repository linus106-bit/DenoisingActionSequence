from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_utils import EOS_ACTION, GridDenoiseDataset, PAD_ACTION
from model import AutoregressiveTrajectoryTransformer, BOS_TOKEN_ID, FlowMatchingTransformer


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

    else:
        raise ValueError(f"Unsupported --model_type: {args.model_type}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            need_debug = args.model_type == "flow_matching" and epoch == 1 and running == 0.0
            if args.model_type == "flow_matching" and need_debug:
                loss, dbg = fm_loss(model, batch, device, return_debug=True, pad_noise_prob=args.pad_noise_prob)
            elif args.model_type == "flow_matching":
                loss = fm_loss(model, batch, device, pad_noise_prob=args.pad_noise_prob)
            else:
                loss = ar_loss(model, batch, device)
            loss.backward()
            if args.model_type == "autoregressive":
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            running += loss.item()

            if need_debug:
                print("[Debug:first-step] t:", round(dbg["t0"], 4))
                print("[Debug:first-step] noisy[0]:", dbg["noisy"].tolist())
                print("[Debug:first-step] clean[0]:", dbg["clean"].tolist())
                print("[Debug:first-step] pred_token(argmax, x0+v)[0]:", _tokenize_for_print(dbg["pred_tokens"]))
                print("[Debug:first-step] token_mse_head(first 10):", dbg["mse_head"].tolist())
                print("[Debug:first-step] loss:", round(dbg["loss"], 6))

        avg = running / max(len(loader), 1)
        print(f"Epoch {epoch}/{args.epochs} - loss: {avg:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": vars(args)}, out)
    print(f"Saved checkpoint: {out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", type=str, choices=["flow_matching", "autoregressive"], default="flow_matching")
    p.add_argument("--n_samples", type=int, default=1500)
    p.add_argument("--grid_size", type=int, default=8)
    p.add_argument("--max_seq_len", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--embed_dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ff_dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--pad_noise_prob", type=float, default=1.0)
    p.add_argument("--out", type=str, default="checkpoints/fm_denoiser.pt")
    return p


if __name__ == "__main__":
    p = build_parser()
    args = p.parse_args()
    train(args)
