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
EMPTY_SLOT = len(BLOCKS)
BLOCKS_TO_PICK = 3

BOARD_FLAT = 81
BOARD_LEN = 9
BOARD_SIZE = (9, 9)

RANDOM_SEED = 13579

NN_OUTPUT_3D = (BLOCKS_TO_PICK, BOARD_LEN, BOARD_LEN)
NN_OUTPUT_FLAT = BLOCKS_TO_PICK * BOARD_FLAT

def _get_valid_placement_mask(board: np.ndarray, idx: int) -> np.ndarray:
    block = BLOCKS[idx]
    valid_mask = np.ones(BOARD_SIZE, dtype=bool)
    n = BOARD_LEN

    for dx, dy in block:
        invalid_positions = board[max(0, dx):min(n, n + dx), 
                                  max(0, dy):min(n, n + dy)]

        current_valid = np.zeros(BOARD_SIZE, dtype=bool)
        current_valid[max(0, -dx):min(n, n - dx), 
                      max(0, -dy):min(n, n - dy)] = ~invalid_positions
        valid_mask &= current_valid

    return valid_mask

class RandomFactory:
    def __init__(self, seed):
        self._seed = seed
        self._random = r.Random(seed)

    def new_blocks(self):
        tmp = self._random.sample(range(BLOCKS_SIZE), BLOCKS_TO_PICK)
        tmp.sort()
        return tmp

    def explore_now(self, eps=0.0):
        return self._random.random() < eps

class State:
    rng = RandomFactory(RANDOM_SEED)

    def __init__(self):
        self._board = np.zeros(BOARD_SIZE, bool)
        self._count = BLOCKS_TO_PICK
        self._available = self.rng.new_blocks()
        self._streak = False
    
    def in_nn_board(self):
        return self._board
    
    def in_nn_pieces(self):
        tmp = self._available.copy()
        for i in range(len(tmp), 3):
            tmp.append(EMPTY_SLOT)
        return tmp

    def in_nn_streak(self):
        return self._streak

    def _calculate_strikes(self, block: list, x: int, y: int):
        streak = False
        rwrd = 0
        # Check for completed rows, columns, and 3x3 squares, note them
        hits = set()
        for dx, dy in block:
            x0 = x + dx
            y0 = y + dy
            hit = False

            if self._board[x0, :].all():
                hits.add((0, x0))
                hit = True
            if self._board[:, y0].all():
                hits.add((1, y0))
                hit = True
            sq_x = (x0 // 3) * 3
            sq_y = (y0 // 3) * 3
            if self._board[sq_x:sq_x + 3, sq_y:sq_y + 3].all():
                hits.add((2, sq_x + sq_y * 8))
                hit = True

            if not hit: 
                rwrd += 1
            else:
                streak = True 

        for what, id in hits:
            if what == 0:
                self._board[id, :] = False
            elif what == 1:
                self._board[:, id] = False
            else:
                org_x = id % 8
                org_y = id // 8
                self._board[org_x:org_x+3, org_y:org_y+3] = False

        rwrd += streak * 9
        rwrd += len(hits) * 9
        return rwrd, streak

    def _calculate_mask(self):
        mask = np.zeros(NN_OUTPUT_3D, bool)
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
        for dx, dy in BLOCKS[block_id]:
            self._board[x + dx, y + dy] = True

        # calculate reward and a new board
        rwrd, self._streak = self._calculate_strikes(block, x, y)

        # check if we need to sample new blocks
        if self._count == 0:
            self._available = self.rng.new_blocks()
            self._count = BLOCKS_TO_PICK

        # check if we are terminal state and return the mask of illegal moves
        mask = self._calculate_mask()
        return rwrd, bool(not mask.any()), mask

    def copy(self):
        return copy.deepcopy(self)

if __name__ == "__main__":
    # x = State()
    # print(x._blocks)
    # print(x._board)
    # rng = RandomFactory()
    # print(rng.new_blocks())
    # print(len(x.neural_input()))
    # print(x.neural_input())

    # for block in BLOCKS:
    #     arr = np.zeros(BOARD_SIZE, np.int8)
    #     for dx, dy in block:
    #         arr[4 + dx, 4 + dy] = 1
    #     print(arr)
    #     print('\n\n')

    st = State()
    print("INITIAL STATE!!")
    print(st._board.astype(int))
    print(st._available)

    print("\n\nNEURAL OUTPUT!!")
    print(st.in_nn_board())
    print(st.in_nn_pieces())

    print("\n\nREWARD, IS_MOVE_FINISHING, MASK_OF_ILLEGAL_MOVES!")
    print(st.transition(0, 4, 4))


    print("\n\nAFTERWARDS STATE!")
    print(st._board.astype(int))

    print("\n\nAFTERWARDS NEURAL OUTPUT")
    print(st.in_nn_board())
    print(st.in_nn_pieces())


del copy
del math
del np
del r