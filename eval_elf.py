from __future__ import annotations

from eval import build_parser, run


if __name__ == "__main__":
    parser = build_parser(default_model_type="elf")
    parser.set_defaults(ckpt="checkpoints/elf_action.pt")
    run(parser.parse_args())
