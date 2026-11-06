"""Plain python / numpy implementation of BlokuDoku rules for a single game.

Deliberately naive and independent of the torch tables in blokudoku_lib
(PLACEMENTS, LINES), so it can serve as an oracle for the vectorized code.
"""

import numpy as np

import blokudoku_lib as bl


def valid_mask(board: np.ndarray, blocks) -> np.ndarray:
    """(9, 9) board, 3 block ids -> (243,) legal moves mask."""
    mask = np.zeros((bl.BLOCKS_TO_PICK, bl.BOARD_LEN, bl.BOARD_LEN), bool)
    for k, b in enumerate(blocks):
        if b == bl.EMPTY_SLOT:
            continue
        for x in range(bl.BOARD_LEN):
            for y in range(bl.BOARD_LEN):
                mask[k, x, y] = all(
                    0 <= x + dx < bl.BOARD_LEN
                    and 0 <= y + dy < bl.BOARD_LEN
                    and not board[x + dx, y + dy]
                    for dx, dy in bl.BLOCKS[b]
                )
    return mask.reshape(-1)


def transition(board: np.ndarray, blocks: np.ndarray, streak: bool, action: int):
    """Apply one move.

    Returns:
        (reward without NEURAL_PENALTY, board, blocks with used slot emptied
        and empty slots moved last, streak). No redraw of blocks is performed.
    """
    board, blocks = board.copy(), blocks.copy()
    piece, x, y = bl.State.decode_action(action)
    cells = [(x + dx, y + dy) for dx, dy in bl.BLOCKS[blocks[piece]]]
    blocks[piece] = bl.EMPTY_SLOT
    empty = blocks == bl.EMPTY_SLOT
    blocks = np.concatenate([blocks[~empty], blocks[empty]])
    for cx, cy in cells:
        board[cx, cy] = True

    hits, reward, new_streak = set(), 0, False
    for cx, cy in cells:
        hit = False
        if board[cx, :].all():
            hits.add(("row", cx))
            hit = True
        if board[:, cy].all():
            hits.add(("col", cy))
            hit = True
        sx, sy = cx // 3 * 3, cy // 3 * 3
        if board[sx : sx + 3, sy : sy + 3].all():
            hits.add(("sq", sx, sy))
            hit = True
        new_streak |= hit
        reward += not hit
    for h in hits:
        if h[0] == "row":
            board[h[1], :] = False
        elif h[0] == "col":
            board[:, h[1]] = False
        else:
            board[h[1] : h[1] + 3, h[2] : h[2] + 3] = False

    reward += (new_streak and streak) * bl.VectorizedStates.STREAK_RWRD
    reward += len(hits) * bl.VectorizedStates.CLEAR_RWRD
    reward += (not board.any()) * bl.VectorizedStates.TOTAL_CLEAR_RWRD
    return reward, board, blocks, new_streak
