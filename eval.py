from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from data_utils import ACTIONS, EOS_ACTION, GridDenoiseDataset, PAD_ACTION
from model import FlowMatchingTransformer


def decode_actions_from_embeddings(
    model: FlowMatchingTransformer, seq_emb: torch.Tensor, mode: str = "argmax"
) -> torch.Tensor:
    # seq_emb: (L, D) -> logits: (L, 6; includes EOS=4 and PAD=5)
    logits = model.action_logits_from_embeddings(seq_emb)
    if mode == "sample":
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    if mode == "argmax":
        return logits.argmax(dim=-1)
    raise ValueError(f"Unsupported decode mode: {mode}")


def rollout(start: Tuple[int, int], actions: List[int], grid):
    pos = start
    traj = [pos]
    h, w = grid.shape
    for a in actions:
        # EOS/PAD are not real actions. Stop rollout when sequence terminates.
        if a in (EOS_ACTION, PAD_ACTION):
            break
        dr, dc = ACTIONS[a]
        nr, nc = pos[0] + dr, pos[1] + dc
        if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0:
            pos = (nr, nc)
        traj.append(pos)
    return traj


def trim_at_stop(actions: List[int]) -> List[int]:
    stop_positions = [actions.index(token) for token in (EOS_ACTION, PAD_ACTION) if token in actions]
    if stop_positions:
        return actions[: min(stop_positions)]
    return actions


def sequence_metrics(pred_tokens: torch.Tensor, clean_tokens: torch.Tensor, clean_valid_len: int) -> dict[str, float]:
    pred_tokens = pred_tokens.cpu()
    clean_tokens = clean_tokens.cpu()
    full_acc = float((pred_tokens == clean_tokens).float().mean().item())
    valid_acc = float((pred_tokens[:clean_valid_len] == clean_tokens[:clean_valid_len]).float().mean().item())
    exact_match = float(torch.equal(pred_tokens, clean_tokens))
    valid_exact_match = float(torch.equal(pred_tokens[:clean_valid_len], clean_tokens[:clean_valid_len]))
    pred_trimmed = trim_at_stop(pred_tokens.tolist())
    clean_trimmed = trim_at_stop(clean_tokens.tolist())
    trimmed_exact_match = float(pred_trimmed == clean_trimmed)
    return {
        "full_token_acc": full_acc,
        "valid_token_acc": valid_acc,
        "exact_match_full": exact_match,
        "exact_match_valid": valid_exact_match,
        "trimmed_exact_match": trimmed_exact_match,
        "pred_len": float(len(pred_trimmed)),
        "clean_len": float(len(clean_trimmed)),
    }


def trajectory_metrics(
    grid, start: Tuple[int, int], goal: Tuple[int, int], actions: List[int]
) -> dict[str, float | int | bool | Tuple[int, int]]:
    traj = rollout(start, actions, grid)
    final_pos = traj[-1]
    return {
        "goal_reached": final_pos == goal,
        "final_pos": final_pos,
        "traj_len": len(traj) - 1,
    }


def print_metrics(tag: str, metrics: dict[str, float | int | bool | Tuple[int, int]]) -> None:
    parts = []
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    print(f"[{tag}] " + ", ".join(parts))


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run(args):
    if args.steps <= 0:
        raise ValueError("--steps must be a positive integer")

    set_seed(args.seed)
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
    noisy_actions = batch["noisy_actions"].unsqueeze(0).to(device)
    valid_len = clean_actions.shape[1]
    clean_valid_len = int((clean_actions[0] != PAD_ACTION).sum().item())

    # The model is trained on action+EOS positions while ignoring PAD.
    mask = (noisy_actions != PAD_ACTION).float()

    with torch.no_grad():
        x0 = model.embed_actions(noisy_actions)

        # One-step denoising
        t0 = torch.zeros((1,), device=device)
        one_step_v = model(x0, t0, map_tensor, mask)
        x_one = x0 + one_step_v
        pred_one = decode_actions_from_embeddings(model, x_one.squeeze(0), mode=args.decode).cpu()

        # Multi-step denoising
        x = x0.clone()
        steps = args.steps
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((1,), i / steps, device=device)
            v = model(x, t, map_tensor, mask)
            x = x + dt * v
            step_pred = decode_actions_from_embeddings(model, x.squeeze(0), mode=args.decode).cpu()[:valid_len].tolist()
            print(f"[step {i + 1:02d}/{steps}] decoded actions: {step_pred}")

        pred = decode_actions_from_embeddings(model, x.squeeze(0), mode=args.decode).cpu()

    noisy_list = trim_at_stop(noisy_actions[0, :valid_len].cpu().tolist())
    one_step_list = trim_at_stop(pred_one[:valid_len].tolist())
    clean_list = trim_at_stop(clean_actions[0, :valid_len].cpu().tolist())
    pred_list = trim_at_stop(pred[:valid_len].tolist())

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

    noisy_seq_metrics = sequence_metrics(noisy_actions[0, :valid_len].cpu(), clean_actions[0, :valid_len].cpu(), clean_valid_len)
    one_step_seq_metrics = sequence_metrics(pred_one[:valid_len], clean_actions[0, :valid_len].cpu(), clean_valid_len)
    pred_seq_metrics = sequence_metrics(pred[:valid_len], clean_actions[0, :valid_len].cpu(), clean_valid_len)
    noisy_traj_metrics = trajectory_metrics(wall, start_cell, goal_cell, noisy_list)
    one_step_traj_metrics = trajectory_metrics(wall, start_cell, goal_cell, one_step_list)
    pred_traj_metrics = trajectory_metrics(wall, start_cell, goal_cell, pred_list)

    print(f"Saved visualization to: {out}")
    print(f"Decode mode: {args.decode}")
    print("Noisy:", noisy_list)
    print("OneStep:", one_step_list)
    print("Pred :", pred_list)
    print("Clean:", clean_list)
    print_metrics("NoisySequence", noisy_seq_metrics)
    print_metrics("OneStepSequence", one_step_seq_metrics)
    print_metrics("PredSequence", pred_seq_metrics)
    print_metrics("NoisyTrajectory", noisy_traj_metrics)
    print_metrics("OneStepTrajectory", one_step_traj_metrics)
    print_metrics("PredTrajectory", pred_traj_metrics)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/fm_denoiser.pt")
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--max_seq_len", type=int, default=None)
    p.add_argument("--plot_out", type=str, default="artifacts/denoise_demo.png")
    p.add_argument("--decode", type=str, choices=["argmax", "sample"], default="argmax")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args)
