from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset

# Actions: 0=Up, 1=Down, 2=Left, 3=Right
PAD_ACTION = -1
ACTIONS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}
OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}


@dataclass
class GridSample:
    grid: np.ndarray  # (10, 10), 0 free / 1 wall
    start: Tuple[int, int]
    goal: Tuple[int, int]
    clean_actions: List[int]
    noisy_actions: List[int]


def _build_graph(grid: np.ndarray) -> nx.Graph:
    h, w = grid.shape
    g = nx.Graph()
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 1:
                continue
            g.add_node((r, c))
            for dr, dc in ACTIONS.values():
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0:
                    g.add_edge((r, c), (nr, nc))
    return g


def _path_to_actions(path: Sequence[Tuple[int, int]]) -> List[int]:
    actions: List[int] = []
    deltas = {v: k for k, v in ACTIONS.items()}
    for (r1, c1), (r2, c2) in zip(path[:-1], path[1:]):
        actions.append(deltas[(r2 - r1, c2 - c1)])
    return actions


def _add_noise(clean_actions: Sequence[int], max_len: int, replace_p: float = 0.25, insert_p: float = 0.2) -> List[int]:
    noisy: List[int] = []
    for a in clean_actions:
        if random.random() < replace_p:
            a = random.randint(0, 3)
        noisy.append(a)
        if random.random() < insert_p and len(noisy) + 2 <= max_len:
            b = random.randint(0, 3)
            noisy.extend([b, OPPOSITE[b]])
    return noisy[:max_len]


def sample_grid_with_path(
    size: int = 10,
    wall_ratio_range: Tuple[float, float] = (0.2, 0.3),
    min_path_len: int = 8,
    max_tries: int = 200,
    max_seq_len: int = 40,
) -> GridSample:
    for _ in range(max_tries):
        ratio = random.uniform(*wall_ratio_range)
        grid = (np.random.rand(size, size) < ratio).astype(np.int64)

        free_cells = np.argwhere(grid == 0)
        if len(free_cells) < 2:
            continue

        start_idx, goal_idx = np.random.choice(len(free_cells), size=2, replace=False)
        start = tuple(int(x) for x in free_cells[start_idx])
        goal = tuple(int(x) for x in free_cells[goal_idx])

        g = _build_graph(grid)
        if start not in g or goal not in g:
            continue

        try:
            path = nx.shortest_path(g, source=start, target=goal)
        except nx.NetworkXNoPath:
            continue

        if len(path) - 1 < min_path_len:
            continue

        clean = _path_to_actions(path)[:max_seq_len]
        noisy = _add_noise(clean, max_len=max_seq_len)
        return GridSample(grid=grid, start=start, goal=goal, clean_actions=clean, noisy_actions=noisy)

    raise RuntimeError("Failed to sample a valid grid/path pair")


class GridDenoiseDataset(Dataset):
    def __init__(self, n_samples: int, max_seq_len: int = 40, grid_size: int = 10):
        self.n_samples = n_samples
        self.max_seq_len = max_seq_len
        self.grid_size = grid_size
        self.samples = [sample_grid_with_path(size=grid_size, max_seq_len=max_seq_len) for _ in range(n_samples)]

    def __len__(self) -> int:
        return self.n_samples

    def _pad(self, actions: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        arr = np.full((self.max_seq_len,), fill_value=PAD_ACTION, dtype=np.int64)
        mask = np.zeros((self.max_seq_len,), dtype=np.float32)
        L = min(len(actions), self.max_seq_len)
        arr[:L] = np.array(actions[:L], dtype=np.int64)
        mask[:L] = 1.0
        return arr, mask

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        noisy, mask = self._pad(sample.noisy_actions)
        clean, _ = self._pad(sample.clean_actions)

        map_tensor = np.zeros((3, self.grid_size, self.grid_size), dtype=np.float32)
        map_tensor[0] = sample.grid.astype(np.float32)  # wall map
        map_tensor[1, sample.start[0], sample.start[1]] = 1.0
        map_tensor[2, sample.goal[0], sample.goal[1]] = 1.0

        return {
            "map": torch.from_numpy(map_tensor),
            "noisy_actions": torch.from_numpy(noisy),
            "clean_actions": torch.from_numpy(clean),
            "mask": torch.from_numpy(mask),
        }
