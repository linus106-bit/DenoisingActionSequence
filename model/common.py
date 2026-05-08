from __future__ import annotations

import math

import torch
import torch.nn as nn

EOS_TOKEN_ID = 5
PAD_TOKEN_ID = 6
BOS_TOKEN_ID = 0


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(
            torch.arange(half, device=t.device, dtype=t.dtype) * (-math.log(10000.0) / max(half - 1, 1))
        )
        args = t[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        positions = torch.arange(length, device=device, dtype=dtype)
        half = self.dim // 2
        freq = torch.exp(
            torch.arange(half, device=device, dtype=dtype) * (-math.log(10000.0) / max(half - 1, 1))
        )
        args = positions[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class MapEncoder(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def resolve_warmup_steps(total_steps: int, warmup_steps: int, warmup_ratio: float) -> int:
    """Resolve an explicit or ratio-based warmup length for a training run.

    A negative ``warmup_steps`` value means "derive it from ``warmup_ratio``".
    The result is clamped so at least one step remains for post-warmup decay when
    the run has more than one optimization step.
    """
    total_steps = max(int(total_steps), 1)
    if warmup_steps < 0:
        warmup_steps = int(total_steps * warmup_ratio)
    max_warmup_steps = max(total_steps - 1, 0)
    return max(0, min(int(warmup_steps), max_warmup_steps))


def warmup_cosine_lr_lambda(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    """Return an LR multiplier with linear warmup followed by cosine decay."""
    total_steps = max(int(total_steps), 1)
    warmup_steps = max(0, min(int(warmup_steps), max(total_steps - 1, 0)))
    min_lr_ratio = float(min_lr_ratio)

    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)

    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    warmup_ratio: float,
    min_lr_ratio: float,
) -> tuple[torch.optim.lr_scheduler.LambdaLR, int]:
    """Build the shared warmup + cosine LR scheduler used by every model type."""
    resolved_warmup_steps = resolve_warmup_steps(total_steps, warmup_steps, warmup_ratio)

    def lr_lambda(step: int) -> float:
        return warmup_cosine_lr_lambda(
            step,
            total_steps=total_steps,
            warmup_steps=resolved_warmup_steps,
            min_lr_ratio=min_lr_ratio,
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda), resolved_warmup_steps
