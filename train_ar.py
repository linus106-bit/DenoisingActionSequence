from __future__ import annotations

from train import build_parser, train


if __name__ == "__main__":
    parser = build_parser()
    parser.set_defaults(model_type="autoregressive", out="checkpoints/ar_trajectory.pt")
    args = parser.parse_args()
    train(args)
