import copy
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
BOARD_SIZE = (9, 9)

RANDOM_SEED = 13579
EPS_DEF = 0.01

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
        self._streak = 0
    
    def neural_input(self):
        tmp_board = self._board.reshape(BOARD_FLAT)
        return np.concat((tmp_board, self._blocks)).astype(np.float64)

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

        # check if we are a terminal state

        # check if we need to pick new blocks
        if self._count == 0:
            self._available = self.rng.new_blocks()
            self._blocks[self._available] = True
            self._count = BLOCKS_TO_PICK

    def copy(self):
        return copy.deepcopy(self)

if __name__ == "__main__":
    x = State()
    print(x._blocks)
    print(x._board)
    rng = RandomFactory()
    print(rng.new_blocks())
    print(len(x.neural_input()))

    for block in BLOCKS:
        arr = np.zeros(BOARD_SIZE, np.int8)
        for dx, dy in block:
            arr[4 + dx, 4 + dy] = 1
        print(arr)
        print('\n\n')

del copy
del np
del r