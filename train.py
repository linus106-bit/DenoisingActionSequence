from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_utils import PAD_ACTION, GridDenoiseDataset
from model import PAD_TOKEN_ID, FlowMatchingTransformer


def _tokenize_for_print(tokens: torch.Tensor) -> list[int]:
    arr = tokens.detach().cpu().tolist()
    return [PAD_ACTION if t == PAD_TOKEN_ID else int(t) for t in arr]


def _make_t_scaled_noisy(clean: torch.Tensor, valid_mask: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Build x0 tokens from clean actions by replacing exactly floor(len(actions) * t)
    valid positions with random actions per sample.
    """
    noisy = clean.clone()
    batch = clean.shape[0]
    for i in range(batch):
        valid_idx = torch.nonzero(valid_mask[i] > 0.5, as_tuple=False).squeeze(-1)
        valid_len = int(valid_idx.numel())
        n_replace = int(valid_len * float(t[i].item()))
        if valid_len == 0 or n_replace <= 0:
            continue

        perm = torch.randperm(valid_len, device=clean.device)
        chosen = valid_idx[perm[:n_replace]]
        original = clean[i, chosen]
        # Ensure replacement action differs from original action (0~3).
        delta = torch.randint(1, 4, size=original.shape, device=clean.device)
        noisy[i, chosen] = (original + delta) % 4
    return noisy


def fm_loss(model, batch, device, return_debug: bool = False):
    map_tensor = batch["map"].to(device)
    clean = batch["clean_actions"].to(device)
    valid_mask = batch["mask"].to(device)
    # Train on full max_seq_len positions to enforce fixed-length behavior.
    mask = torch.ones_like(batch["mask"], device=device)

    t = torch.rand(clean.shape[0], device=device)
    noisy = _make_t_scaled_noisy(clean, valid_mask, t)

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


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = GridDenoiseDataset(n_samples=args.n_samples, max_seq_len=args.max_seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = FlowMatchingTransformer(embed_dim=args.embed_dim, n_layers=args.layers, n_heads=args.heads).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            need_debug = epoch == 1 and running == 0.0
            if need_debug:
                loss, dbg = fm_loss(model, batch, device, return_debug=True)
            else:
                loss = fm_loss(model, batch, device)
            loss.backward()
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


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples", type=int, default=1500)
    p.add_argument("--max_seq_len", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--embed_dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--out", type=str, default="checkpoints/fm_denoiser.pt")
    args = p.parse_args()
    train(args)
