from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch

from data_utils import ACTIONS, GridDenoiseDataset, PAD_ACTION
from model import PAD_TOKEN_ID, FlowMatchingTransformer


def decode_actions_from_embeddings(model: FlowMatchingTransformer, seq_emb: torch.Tensor) -> torch.Tensor:
    # seq_emb: (L, D) -> logits: (L, 5; PAD 포함) -> softmax -> multinomial sampling
    logits = model.action_logits_from_embeddings(seq_emb)
    probs = torch.softmax(logits, dim=-1)
    token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
    # Convert PAD token id(4) back to PAD action value(-1)
    return torch.where(token_ids == PAD_TOKEN_ID, torch.full_like(token_ids, PAD_ACTION), token_ids)


def rollout(start: Tuple[int, int], actions: List[int], grid):
    pos = start
    traj = [pos]
    h, w = grid.shape
    for a in actions:
        # PAD(-1) is not a real action. Stop rollout when PAD begins.
        if a == -1:
            break
        dr, dc = ACTIONS[a]
        nr, nc = pos[0] + dr, pos[1] + dc
        if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0:
            pos = (nr, nc)
        traj.append(pos)
    return traj


def trim_at_pad(actions: List[int]) -> List[int]:
    if PAD_ACTION in actions:
        return actions[: actions.index(PAD_ACTION)]
    return actions


def plot_paths(
    grid,
    start,
    goal,
    noisy_actions,
    one_step_actions,
    multi_step_actions,
    clean_actions,
    out_path: Path,
    multi_step_label: str,
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Noisy path", "one step", multi_step_label]
    seqs = [noisy_actions, one_step_actions, multi_step_actions]

    for idx, (ax, title, seq) in enumerate(zip(axes, titles, seqs)):
        ax.imshow(grid, cmap="gray_r")
        traj = rollout(start, seq, grid)
        ys = [p[0] for p in traj]
        xs = [p[1] for p in traj]
        line_color = "orange" if idx == 0 else None
        ax.plot(xs, ys, marker="o", linewidth=2, color=line_color)
        ax.scatter(start[1], start[0], c="lime", s=80, label="start")
        ax.scatter(goal[1], goal[0], c="red", s=80, label="goal")
        ax.set_title(title)
        ax.set_xlim(-0.5, grid.shape[1] - 0.5)
        ax.set_ylim(grid.shape[0] - 0.5, -0.5)
        ax.grid(True, alpha=0.3)

    clean_traj = rollout(start, clean_actions, grid)
    axes[2].plot([p[1] for p in clean_traj], [p[0] for p in clean_traj], "--", color="gold", label="clean gt")
    axes[2].legend(loc="upper right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["cfg"]
    model = FlowMatchingTransformer(
        embed_dim=cfg["embed_dim"], n_layers=cfg["layers"], n_heads=cfg["heads"]
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    max_seq_len = args.max_seq_len if args.max_seq_len is not None else cfg["max_seq_len"]
    ds = GridDenoiseDataset(n_samples=1, max_seq_len=max_seq_len)
    batch = ds[0]

    map_tensor = batch["map"].unsqueeze(0).to(device)
    clean_actions = batch["clean_actions"].unsqueeze(0).to(device)
    # Generation mode: model should produce a sequence within max_seq_len from map only.
    # So we use full-length mask and random initial actions for the entire sequence.
    mask = torch.ones((1, max_seq_len), device=device)

    noisy_actions = torch.randint(0, 4, size=(1, max_seq_len), device=device)

    with torch.no_grad():
        x0 = model.embed_actions(noisy_actions)

        # One-step denoising
        t0 = torch.zeros((1,), device=device)
        one_step_v = model(x0, t0, map_tensor, mask)
        x_one = x0 + one_step_v
        pred_one = decode_actions_from_embeddings(model, x_one.squeeze(0)).cpu()

        # Multi-step denoising
        x = x0.clone()
        steps = args.steps
        dt = 1.0 / steps
        valid_len = max_seq_len
        for i in range(steps):
            t = torch.full((1,), i / steps, device=device)
            v = model(x, t, map_tensor, mask)
            x = x + dt * v
            step_pred = decode_actions_from_embeddings(model, x.squeeze(0)).cpu()[:valid_len].tolist()
            print(f"[step {i + 1:02d}/{steps}] decoded actions: {step_pred}")

        pred = decode_actions_from_embeddings(model, x.squeeze(0)).cpu()

    noisy_list = trim_at_pad(noisy_actions[0, :valid_len].cpu().tolist())
    one_step_list = trim_at_pad(pred_one[:valid_len].tolist())
    clean_list = trim_at_pad(clean_actions[0, :valid_len].cpu().tolist())
    pred_list = trim_at_pad(pred[:valid_len].tolist())

    wall = batch["map"][0].numpy()
    start_cell = tuple(torch.nonzero(batch["map"][1], as_tuple=False)[0].tolist())
    goal_cell = tuple(torch.nonzero(batch["map"][2], as_tuple=False)[0].tolist())

    out = Path(args.plot_out)
    plot_paths(
        wall,
        start_cell,
        goal_cell,
        noisy_list,
        one_step_list,
        pred_list,
        clean_list,
        out,
        multi_step_label=f"{args.steps} step",
    )
    print(f"Saved visualization to: {out}")
    print("Noisy:", noisy_list)
    print("OneStep:", one_step_list)
    print("Pred :", pred_list)
    print("Clean:", clean_list)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/fm_denoiser.pt")
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--max_seq_len", type=int, default=None)
    p.add_argument("--plot_out", type=str, default="artifacts/denoise_demo.png")
    args = p.parse_args()
    run(args)
