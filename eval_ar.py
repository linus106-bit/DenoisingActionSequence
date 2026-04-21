from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data_utils import EOS_ACTION, GridDenoiseDataset, PAD_ACTION
from eval import aggregate_numeric_metrics, make_json_safe, print_metrics, sequence_metrics, trajectory_metrics
from model import AutoregressiveTrajectoryTransformer


def trim_at_stop(actions: list[int]) -> list[int]:
    stop_positions = [actions.index(token) for token in (EOS_ACTION, PAD_ACTION) if token in actions]
    if stop_positions:
        return actions[: min(stop_positions)]
    return actions


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
    grid_size = args.grid_size if args.grid_size is not None else cfg.get("grid_size", 10)
    ds = GridDenoiseDataset(
        n_samples=args.num_eval_samples,
        max_seq_len=max_seq_len,
        grid_size=grid_size,
    )

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
        print_metrics(f"Sample{sample_idx:02d}PredSequence", pred_seq_metrics)
        print_metrics(f"Sample{sample_idx:02d}PredTrajectory", pred_traj_metrics)

        sample_results.append(
            {
                "sample_idx": sample_idx,
                "start_cell": list(start_cell),
                "goal_cell": list(goal_cell),
                "pred_actions": trim_at_stop(pred.tolist()),
                "clean_actions": trim_at_stop(clean_actions.tolist()),
                "pred_sequence_metrics": make_json_safe(pred_seq_metrics),
                "pred_trajectory_metrics": make_json_safe(pred_traj_metrics),
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
    p.add_argument("--decode", type=str, choices=["argmax", "sample"], default="argmax")
    p.add_argument("--temperature", type=float, default=1.0)
    args = p.parse_args()
    run(args)
