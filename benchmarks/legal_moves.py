"""Benchmark legal_moves (one big matmul) vs legal_moves2 (bmm per game).

Run: uv run python benchmarks/legal_moves.py
"""

import torch
from torch.utils import benchmark

import blokudoku_lib as bl


def random_states(n: int, fill: float = 0.3) -> bl.VectorizedStates:
    vs = bl.VectorizedStates(n)
    vs.boards = torch.rand((n, bl.BOARD_FLAT), device=bl.DEVICE) < fill
    vs.available = torch.randint(
        0,
        bl.BLOCKS_SIZE + 1,
        (n, bl.BLOCKS_TO_PICK),
        device=bl.DEVICE,
        dtype=torch.uint8,
    )
    return vs


if __name__ == "__main__":
    results = []
    for n in (1, 5, 64, 512, 4096, bl.VectorizedStates.MAX_BOARDS):
        vs = random_states(n)
        for name in ("legal_moves", "legal_moves2"):
            results.append(
                benchmark.Timer(
                    stmt=f"vs.{name}()",
                    globals={"vs": vs},
                    label=f"Legal moves ({bl.DEVICE})",
                    sub_label=f"N={n}",
                    description=name,
                ).blocked_autorange(min_run_time=0.5)
            )
    benchmark.Compare(results).print()
