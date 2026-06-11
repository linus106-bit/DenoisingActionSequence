from __future__ import annotations

from train import apply_model_size_preset, build_parser, train


if __name__ == "__main__":
    parser = build_parser()
    parser.set_defaults(model_type="elf", out="checkpoints/elf_action.pt")
    args = parser.parse_args()
    args = apply_model_size_preset(args)
    train(args)
