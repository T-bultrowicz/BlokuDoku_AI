import numpy as np
import pytest
import torch

import blokudoku_lib as bl


@pytest.fixture(autouse=True)
def seed():
    """Every test starts from the same RNG state."""
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture
def random_states():
    """Factory of VectorizedStates with random boards, blocks and streaks."""

    def make(n: int, fill: float = 0.3, valid: bool = False) -> bl.VectorizedStates:
        """valid=True removes full lines (they never survive a real move) and
        makes some lines almost full, so clearings happen often."""
        vs = bl.VectorizedStates(n)
        boards = np.random.rand(n, bl.BOARD_FLAT) < fill
        if valid:
            lines = bl.VectorizedStates.LINES.cpu().numpy()
            for g in range(n):
                for line in np.random.choice(len(lines), 2, replace=False):
                    cells = np.flatnonzero(lines[line])
                    boards[g, cells] = True
                    boards[g, np.random.choice(cells)] = False
                # Clearing a cell cannot complete a line, so one pass suffices.
                for line in range(len(lines)):
                    cells = np.flatnonzero(lines[line])
                    if boards[g, cells].all():
                        boards[g, np.random.choice(cells)] = False
            assert not (
                (boards.astype(int) @ lines.T.astype(int)) == bl.BOARD_LEN
            ).any()
        vs.boards = bl.to_tensor(boards, torch.bool)
        vs.available = bl.to_tensor(
            np.random.randint(0, bl.BLOCKS_SIZE + 1, (n, bl.BLOCKS_TO_PICK)),
            torch.uint8,
        )
        vs.streaks = bl.to_tensor(np.random.rand(n, 1) < 0.5, torch.bool)
        return vs

    return make


def to_numpy(*tensors):
    return tuple(t.cpu().numpy() for t in tensors)
