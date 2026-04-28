from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from data_utils import EOS_ACTION, GridDenoiseDataset, PAD_ACTION
from eval import (
    aggregate_numeric_metrics,
    make_json_safe,
    print_metrics,
    rollout,
    sequence_metrics,
    trajectory_metrics,
)
from model import AutoregressiveTrajectoryTransformer


def trim_at_stop(actions: list[int]) -> list[int]:
    stop_positions = [actions.index(token) for token in (EOS_ACTION, PAD_ACTION) if token in actions]
    if stop_positions:
        return actions[: min(stop_positions)]
    return actions


def plot_pred_vs_clean(
    wall,
    start_cell: tuple[int, int],
    goal_cell: tuple[int, int],
    clean_actions: list[int],
    pred_actions: list[int],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    titles_and_actions = [("Clean path", clean_actions), ("AR predicted path", pred_actions)]
    for ax, (title, actions) in zip(axes, titles_and_actions):
        ax.imshow(wall, cmap="gray_r")
        traj = rollout(start_cell, actions, wall)
        ys = [p[0] for p in traj]
        xs = [p[1] for p in traj]
        ax.plot(xs, ys, marker="o", linewidth=2)
        ax.scatter(start_cell[1], start_cell[0], c="lime", s=80, label="start")
        ax.scatter(goal_cell[1], goal_cell[0], c="red", s=80, label="goal")
        ax.set_title(title)
        ax.set_xlim(-0.5, wall.shape[1] - 0.5)
        ax.set_ylim(wall.shape[0] - 0.5, -0.5)
        ax.grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["cfg"]
    model = AutoregressiveTrajectoryTransformer(
        embed_dim=cfg["embed_dim"], n_layers=cfg["layers"], n_heads=cfg["heads"]
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    max_seq_len = args.max_seq_len if args.max_seq_len is not None else cfg["max_seq_len"]
    grid_size = args.grid_size if args.grid_size is not None else cfg.get("grid_size", 8)
    ds = GridDenoiseDataset(
        n_samples=args.num_eval_samples,
        max_seq_len=max_seq_len,
        grid_size=grid_size,
    )
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    sample_results: list[dict] = []
    for sample_idx in range(args.num_eval_samples):
        batch = ds[sample_idx]
        map_tensor = batch["map"].unsqueeze(0).to(device)
        clean_actions = batch["clean_actions"][:max_seq_len].cpu()
        clean_valid_len = int((clean_actions != PAD_ACTION).sum().item())
        with torch.no_grad():
            pred = model.generate(
                map_tensor,
                max_len=max_seq_len,
                decode=args.decode,
                temperature=args.temperature,
            )[0].cpu()
        pred_seq_metrics = sequence_metrics(pred, clean_actions, clean_valid_len)
        wall = batch["map"][0].numpy()
        start_cell = tuple(torch.nonzero(batch["map"][1], as_tuple=False)[0].tolist())
        goal_cell = tuple(torch.nonzero(batch["map"][2], as_tuple=False)[0].tolist())
        pred_traj_metrics = trajectory_metrics(wall, start_cell, goal_cell, trim_at_stop(pred.tolist()))
        plot_path = plot_dir / f"sample_{sample_idx:02d}.png"
        plot_pred_vs_clean(
            wall=wall,
            start_cell=start_cell,
            goal_cell=goal_cell,
            clean_actions=trim_at_stop(clean_actions.tolist()),
            pred_actions=trim_at_stop(pred.tolist()),
            out_path=plot_path,
        )
        print_metrics(f"Sample{sample_idx:02d}PredSequence", pred_seq_metrics)
        print_metrics(f"Sample{sample_idx:02d}PredTrajectory", pred_traj_metrics)
        print(f"[sample {sample_idx:02d}] plot={plot_path}")

        sample_results.append(
            {
                "sample_idx": sample_idx,
                "start_cell": list(start_cell),
                "goal_cell": list(goal_cell),
                "pred_actions": trim_at_stop(pred.tolist()),
                "clean_actions": trim_at_stop(clean_actions.tolist()),
                "pred_sequence_metrics": make_json_safe(pred_seq_metrics),
                "pred_trajectory_metrics": make_json_safe(pred_traj_metrics),
                "plot_path": str(plot_path),
            }
        )

    summary = {
        "ckpt": args.ckpt,
        "decode": args.decode,
        "temperature": args.temperature,
        "num_eval_samples": args.num_eval_samples,
        "grid_size": grid_size,
        "aggregate_pred_metrics": aggregate_numeric_metrics(sample_results),
        "samples": sample_results,
    }
    out = Path(args.results_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"Saved results to: {out}")
    print_metrics("AggregatePred", summary["aggregate_pred_metrics"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/ar_trajectory.pt")
    p.add_argument("--grid_size", type=int, default=None)
    p.add_argument("--max_seq_len", type=int, default=None)
    p.add_argument("--num_eval_samples", type=int, default=10)
    p.add_argument("--results_out", type=str, default="artifacts/eval_ar_results.json")
    p.add_argument("--plot_dir", type=str, default="artifacts/eval_ar_plots")
    p.add_argument("--decode", type=str, choices=["argmax", "sample"], default="argmax")
    p.add_argument("--temperature", type=float, default=1.0)
    args = p.parse_args()
    run(args)
