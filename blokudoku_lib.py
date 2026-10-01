import copy
import math
import numpy as np
import random as r

BLOCKS = [
    # SQUARE PIECES
    [(0, 0)], 
    [(0, 0), (1, 0), (0, 1), (1, 1)],
    # LINE PIECES - LATERAL
    [(0, 0), (1, 0)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 0), (1, 0), (2, 0), (3, 0)],
    [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
    # LINE PIECES - VERTICAL
    [(0, 0), (0, 1)],
    [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (0, 1), (0, 2), (0, 3)],
    [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
    # LINE PIECES - DIAGONAL UPWARD
    [(0, 0), (1, 1)],
    [(0, 0), (1, 1), (2, 2)],
    # LINE PIECES - DIAGONAL DOWNWARD
    [(0, 0), (-1, 1)],
    [(0, 0), (-1, 1), (-2, 2)],
    # L-SHORT PIECES
    [(0, 0), (0, 1), (-1, 0)],
    [(0, 0), (0, 1), (1, 0)],
    [(0, 0), (0, 1), (-1, 1)],
    [(0, 0), (0, 1), (1, 1)],
    # L-LONG PIECES
    [(0, 0), (0, 1), (0, 2), (-1, 0)],
    [(0, 0), (0, 1), (0, 2), (1, 0)],
    [(0, 0), (0, 1), (0, 2), (-1, 2)],
    [(0, 0), (0, 1), (0, 2), (1, 2)],
    [(0, 0), (1, 0), (2, 0), (0, 1)],
    [(0, 0), (1, 0), (2, 0), (0, -1)],
    [(0, 0), (1, 0), (2, 0), (2, 1)],
    [(0, 0), (1, 0), (2, 0), (2, -1)],
    # L -ULTRA LONG PIECES
    [(0, 0), (0, 1), (0, 2), (-1, 0), (-2, 0)],
    [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
    [(0, 0), (0, 1), (0, 2), (-1, 2), (-2, 2)],
    [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)],
    # T-SHORT PIECES
    [(0, 0), (1, 0), (2, 0), (1, 1)],
    [(0, 0), (1, 0), (2, 0), (1, -1)],
    [(0, 0), (0, 1), (0, 2), (-1, 1)],
    [(0, 0), (0, 1), (0, 2), (1, 1)],
    # T-LONG PIECES
    [(0, 0), (1, 0), (2, 0), (1, 1), (1, 2)],
    [(0, 0), (1, 0), (2, 0), (1, -1), (1, -2)],
    [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)],
    [(0, 0), (1, 0), (2, 0), (2, -1), (2, 1)], 
    # U PIECES
    [(0, 0), (1, 0), (0, 2), (1, 2), (1, 1)],
    [(0, 0), (1, 0), (0, 2), (1, 2), (0, 1)],
    [(0, 0), (1, 0), (2, 0), (2, 1), (0, 1)],
    [(0, 0), (1, 1), (2, 0), (2, 1), (0, 1)],
    # PLUS PIECE
    [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)],
]
BLOCKS_SIZE = len(BLOCKS)
BLOCKS_TO_PICK = 3

BOARD_FLAT = 81
BOARD_LEN = 9
BOARD_SIZE = (9, 9)

RANDOM_SEED = 13579
EPS_DEF = 0.01

NN_INPUT_SIZE = BOARD_FLAT + BLOCKS_SIZE + 1
NN_OUTPUT_3D = (BLOCKS_TO_PICK, BOARD_LEN, BOARD_LEN)
NN_OUTPUT_FLAT = BLOCKS_TO_PICK * BOARD_FLAT

def _get_valid_placement_mask(board: np.ndarray, idx: int) -> np.ndarray:
    block = BLOCKS[idx]
    valid_mask = np.ones(BOARD_SIZE, dtype=np.bool)
    n = BOARD_LEN

    for dx, dy in block:
        invalid_positions = board[max(0, dx):min(n, n + dx), 
                                  max(0, dy):min(n, n + dy)]

        current_valid = np.zeros(BOARD_SIZE, dtype=np.bool)
        current_valid[max(0, -dx):min(n, n - dx), 
                      max(0, -dy):min(n, n - dy)] = ~invalid_positions
        valid_mask &= current_valid

    return valid_mask


class RandomFactory:
    def __init__(self, seed=RANDOM_SEED):
        self._seed = seed
        self._random = r.Random(seed)

    def new_blocks(self):
        return r.sample(range(BLOCKS_SIZE), BLOCKS_TO_PICK)

    def explore_now(self, eps=EPS_DEF):
        return self._random.random() < eps

class State:
    rng = RandomFactory()

    def __init__(self):
        self._board = np.zeros(BOARD_SIZE, np.bool)
        self._blocks = np.zeros(BLOCKS_SIZE, np.bool)
        self._count = BLOCKS_TO_PICK

        self._available = self.rng.new_blocks()
        self._blocks[self._available] = True
        self._streak = False
    
    def neural_input(self):
        tmp_board = self._board.reshape(BOARD_FLAT)
        return np.concat((
            tmp_board, 
            self._blocks, 
            [self._streak])
        ).astype(np.float32)

    def _calculate_strikes(self, block: list, x: int, y: int):
        rwrd = 9 if self._streak else 0
        streak = False

        set_xs = {x + dx for dx, _ in block}
        set_ys = {y + dy for _, dy in block}
        set_squares = set()
        for dx, dy in block:
            set_squares.add(((x + dx) / 3, (y + dy) / 3))

        for xs in set_xs:
            if self._board[xs, :].all():
                rwrd += 9
                self._board[xs, :] = False
                streak = True

        for ys in set_ys:
            if self._board[:, ys].all():
                rwrd += 9
                self._board[:, ys] = False
                streak = True

        for x0, y0 in set_squares:
            if self._board[x0:x0 + 3, y0:y0 + 3].all():
                rwrd += 9
                self._board[x0:x0 + 3, y0:y0 + 3] = False
                streak = True

        return rwrd, streak

    def _calculate_mask(self):
        mask = np.zeros(NN_OUTPUT_3D, np.bool)
        mask[0, :, :] = _get_valid_placement_mask(self._board, self._available[0])

        if self._count == 1:
            mask[1, :, :] = False
            mask[2, :, :] = False
        elif self._count == 2:
            mask[1, :, :] = _get_valid_placement_mask(self._board, self._available[1])
            mask[2, :, :] = False
        elif self._count == 3:
            mask[1, :, :] = _get_valid_placement_mask(self._board, self._available[1])
            mask[2, :, :] = _get_valid_placement_mask(self._board, self._available[2])
        else:
            raise ValueError(f"_count value: {self._count} is not in range 1-3!")

        return mask.reshape(NN_OUTPUT_FLAT)

    def transition(self, piece: int, x: int, y: int):
        block_id = self._available[piece]
        block = BLOCKS[block_id]

        # make move
        self._count -= 1
        self._available.remove(block_id)
        self._blocks[block_id] = False
        for dx, dy in BLOCKS[block_id]:
            self._board[x + dx, y + dy] = True

        # calculate reward and a new board
        rwrd, self._streak = self._calculate_strikes(block, x, y)

        # check if we need to pick new blocks
        if self._count == 0:
            self._available = self.rng.new_blocks()
            self._blocks[self._available] = True
            self._count = BLOCKS_TO_PICK

        # check if we are terminal state and return the mask of illegal moves
        mask = self._calculate_mask()
        return rwrd, bool(not mask.any()), mask

    def copy(self):
        return copy.deepcopy(self)

if __name__ == "__main__":
    x = State()
    print(x._blocks)
    print(x._board)
    rng = RandomFactory()
    print(rng.new_blocks())
    print(len(x.neural_input()))
    print(x.neural_input())

    for block in BLOCKS:
        arr = np.zeros(BOARD_SIZE, np.int8)
        for dx, dy in block:
            arr[4 + dx, 4 + dy] = 1
        print(arr)
        print('\n\n')

del copy
del math
del np
del r