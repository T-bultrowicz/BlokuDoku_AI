"""Scoring rules checked on hand-built boards, one rule per test.

Rules under test (reward of a single move, NEURAL_PENALTY aside):
  +1   per placed cell that is still taken after clearings,
  +18  per cleared row / column / 3x3 square,
  +9   when this move clears and the previous move cleared too (once per move),
  +100 when the board is empty after the move,
  -50  (NEURAL_PENALTY) when no block fits after the move.
"""

import numpy as np
import pytest
import torch

import blokudoku_lib as bl
from blokudoku_lib import (
    BLOCKS,
    BLOCKS_SIZE,
    BOARD_FLAT,
    BOARD_LEN,
    EMPTY_SLOT,
    NEURAL_PENALTY,
    VectorizedStates,
)

CELL = 0  # single cell block
PAIR = 2  # vertical pair: (x, y), (x + 1, y)
LINE3 = 3  # vertical line of 3 cells
LINE5 = 5  # vertical line of 5 cells
CLEAR = VectorizedStates.CLEAR_RWRD
STREAK = VectorizedStates.STREAK_RWRD
TOTAL = VectorizedStates.TOTAL_CLEAR_RWRD


def board_of(*cells) -> np.ndarray:
    """(9, 9) bool board with given (x, y) cells taken."""
    board = np.zeros((BOARD_LEN, BOARD_LEN), bool)
    for x, y in cells:
        board[x, y] = True
    return board


def row(x, skip=()):
    return [(x, y) for y in range(BOARD_LEN) if y not in skip]


def col(y, skip=()):
    return [(x, y) for x in range(BOARD_LEN) if x not in skip]


def square(sx, sy, skip=()):
    return [
        (x, y)
        for x in range(sx, sx + 3)
        for y in range(sy, sy + 3)
        if (x, y) not in skip
    ]


def play(board: np.ndarray, blocks, streak: bool, x: int, y: int):
    """Play blocks[0] at (x, y) on a single game.

    Returns:
        (reward, board after as (9, 9) bools, streak after, terminal).
    """
    vs = VectorizedStates(1)
    vs.boards = bl.to_tensor(board.reshape(1, BOARD_FLAT), torch.bool)
    vs.available = bl.to_tensor(np.array([blocks]), torch.uint8)
    vs.streaks = bl.to_tensor(np.array([[streak]]), torch.bool)
    A, R, T, B, _, S, _ = vs.states_prim()
    i = (A == bl.State.encode_action(0, x, y)).nonzero().flatten()
    assert len(i) == 1, "move must be legal"
    board_after = B[i].cpu().numpy().reshape(BOARD_LEN, BOARD_LEN)
    return int(R[i]), board_after, bool(S[i, 0]), bool(T[i])


# Other slots hold single cells, so the game never ends on a non-full board.
SAFE = [CELL, CELL]
FILLER = (4, 4)  # a lonely cell that keeps the board non-empty


class TestPlacedCells:
    @pytest.mark.parametrize("block", range(BLOCKS_SIZE))
    def test_one_point_per_cell_on_empty_board(self, block):
        pos = int(VectorizedStates.IN_BOUNDS[block].nonzero()[0])
        reward, board, streak, terminal = play(
            board_of(), [block, *SAFE], False, pos // BOARD_LEN, pos % BOARD_LEN
        )
        assert reward == len(BLOCKS[block])
        assert board.sum() == len(BLOCKS[block])
        assert not streak and not terminal

    def test_cells_surviving_a_clearing_still_count(self):
        # PAIR at (0, 8) completes row 0, its second cell (1, 8) stays.
        reward, board, _, _ = play(
            board_of(*row(0, skip=[8]), FILLER), [PAIR, *SAFE], False, 0, 8
        )
        assert reward == CLEAR + 1
        assert np.array_equal(board, board_of((1, 8), FILLER))

    def test_cells_removed_by_a_clearing_do_not_count(self):
        reward, _, _, _ = play(
            board_of(*row(0, skip=[8]), FILLER), [CELL, *SAFE], False, 0, 8
        )
        assert reward == CLEAR


class TestClearings:
    @pytest.mark.parametrize(
        "cells, move",
        [
            (row(3, skip=[8]), (3, 8)),
            (col(2, skip=[0]), (0, 2)),
            (square(0, 0, skip=[(2, 2)]), (2, 2)),
        ],
    )
    def test_single_line_is_18(self, cells, move):
        """Rows, columns and squares all clear for 18 and leave the rest."""
        reward, board, streak, _ = play(
            board_of(*cells, FILLER), [CELL, *SAFE], False, *move
        )
        assert reward == CLEAR
        assert np.array_equal(board, board_of(FILLER))
        assert streak

    def test_row_and_column_at_once_is_36(self):
        cells = row(0, skip=[8]) + col(8, skip=[0])
        reward, board, _, _ = play(board_of(*cells, FILLER), [CELL, *SAFE], False, 0, 8)
        assert reward == 2 * CLEAR
        assert np.array_equal(board, board_of(FILLER))

    def test_row_column_and_square_at_once_is_54(self):
        cells = row(0, skip=[0]) + col(0, skip=[0]) + square(0, 0, skip=[(0, 0)])
        reward, board, _, _ = play(board_of(*cells, FILLER), [CELL, *SAFE], False, 0, 0)
        assert reward == 3 * CLEAR
        assert np.array_equal(board, board_of(FILLER))

    def test_three_rows_at_once_with_total_clear(self):
        """Rows 2, 3, 4 miss column 0 only; rows 3-5 or 0-2 would already
        form full squares, which never survive a real move."""
        cells = [c for x in (2, 3, 4) for c in row(x, skip=[0])]
        reward, board, _, _ = play(board_of(*cells), [LINE3, *SAFE], False, 2, 0)
        assert reward == 3 * CLEAR + TOTAL
        assert not board.any()


class TestStreak:
    ROW = board_of(*row(0, skip=[8]), FILLER)

    def test_no_bonus_when_previous_move_did_not_clear(self):
        reward, _, streak, _ = play(self.ROW, [CELL, *SAFE], False, 0, 8)
        assert reward == CLEAR
        assert streak

    def test_bonus_when_previous_move_cleared(self):
        reward, _, streak, _ = play(self.ROW, [CELL, *SAFE], True, 0, 8)
        assert reward == CLEAR + STREAK
        assert streak

    def test_bonus_counted_once_for_double_clearing(self):
        cells = row(0, skip=[8]) + col(8, skip=[0])
        reward, _, streak, _ = play(board_of(*cells, FILLER), [CELL, *SAFE], True, 0, 8)
        assert reward == 2 * CLEAR + STREAK
        assert streak

    def test_streak_lost_on_move_without_clearing(self):
        reward, _, streak, _ = play(board_of(FILLER), [CELL, *SAFE], True, 0, 0)
        assert reward == 1
        assert not streak

    def test_streak_flag_alone_gives_nothing(self):
        """Streak flag True on a move without clearing adds no points at all."""
        with_flag, _, _, _ = play(board_of(FILLER), [PAIR, *SAFE], True, 0, 0)
        without, _, _, _ = play(board_of(FILLER), [PAIR, *SAFE], False, 0, 0)
        assert with_flag == without == 2


class TestTotalClear:
    def test_empty_board_after_move_is_100_extra(self):
        reward, board, _, terminal = play(
            board_of(*row(0, skip=[8])), [CELL, *SAFE], False, 0, 8
        )
        assert reward == CLEAR + TOTAL
        assert not board.any() and not terminal

    def test_total_clear_with_streak(self):
        reward, _, _, _ = play(board_of(*row(0, skip=[8])), [CELL, *SAFE], True, 0, 8)
        assert reward == CLEAR + STREAK + TOTAL

    def test_empty_board_only_when_really_empty(self):
        reward, _, _, _ = play(
            board_of(*row(0, skip=[8]), FILLER), [CELL, *SAFE], False, 0, 8
        )
        assert reward == CLEAR


class TestTerminalPenalty:
    # Full board with holes: the diagonal, (0, 1), (1, 0) and one hole in every
    # off-diagonal square. No line is full, no line completes after a cell at
    # (0, 0), and no vertical 5-line fits anywhere.
    HOLES = [(i, i) for i in range(BOARD_LEN)] + [(0, 1), (1, 0)]
    HOLES += [(0, 4), (0, 7), (3, 1), (3, 7), (6, 1), (6, 4)]
    FULL = ~board_of(*HOLES)

    def test_penalty_when_remaining_blocks_do_not_fit(self):
        reward, _, _, terminal = play(self.FULL, [CELL, LINE5, LINE5], False, 0, 0)
        assert terminal
        assert reward == 1 + NEURAL_PENALTY

    def test_no_penalty_when_a_block_still_fits(self):
        reward, _, _, terminal = play(self.FULL, [CELL, LINE5, CELL], False, 0, 0)
        assert not terminal
        assert reward == 1


class TestRewardBounds:
    """Sanity bounds that make absurd per-move rewards impossible."""

    def test_random_states_rewards_within_rule_bounds(self, random_states):
        vs = random_states(300, 0.5, valid=True)
        _, R, T, _, _, _, _ = vs.states_prim()
        R = R.cpu().numpy().astype(int) - np.where(T.cpu().numpy(), NEURAL_PENALTY, 0)
        # Up to 5 placed cells; a 5-cell line can clear at most 5 + 1 + 2 lines.
        assert R.min() >= 0
        assert R.max() <= 5 + 8 * CLEAR + STREAK + TOTAL
