import copy
import math
import numpy as np
import random as rand

# Array defining shapes of all possible blocks with rotations.
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
    # CHAIR PIECES
    [(0, 0), (0, 1), (1, 1), (1, 2)],
    [(0, 0), (0, 1), (-1, 1), (-1, 2)],
    [(0, 0), (1, 0), (0, 1), (-1, 1)],
    [(0, 0), (-1, 0), (0, 1), (1, 1)],
    # PLUS PIECE
    [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)],
]
BLOCKS_SIZE = len(BLOCKS)
EMPTY_SLOT = len(BLOCKS)  # Descripes empty block choice.
BLOCKS_TO_PICK = 3
BOARD_FLAT = 81
BOARD_LEN = 9
BOARD_SIZE = (9, 9)
NN_OUTPUT_FLAT = BLOCKS_TO_PICK * BOARD_FLAT

# Parameter for NN training, used to punish terminal states.
NEURAL_PENALTY = -50


# Factory used to generate states in BlokuDoku training games.
class RandomFactory:
    DEF_SEED = 13579

    def __init__(self, seed=None):
        if seed is None:
            self._random = rand.Random()
        else:
            self._random = rand.Random(seed)

    def new_blocks(self):
        """Return new list of random 3 blocks."""
        return self._random.choices(range(BLOCKS_SIZE), k=BLOCKS_TO_PICK)

    def sample(self, list, k: int = 1):
        return self._random.sample(list, k)

    def choice(self, list):
        return self._random.choice(list)

    def explore_now(self, eps: float = 0.0):
        """Draw whether agent of given eps should perform exploration."""
        return self._random.random() < eps


rng_fact_instance = RandomFactory(RandomFactory.DEF_SEED)


class State:
    TOTAL_CLEANUP_REWARD = 100
    CLEARING_REWARD = 18
    STREAK_REWARD = 9

    def __init__(self):
        """Used to encapsulate all logic of Bloku Doku game."""
        self._board = np.zeros(BOARD_SIZE, bool)
        self._count = BLOCKS_TO_PICK
        self._available = rng_fact_instance.new_blocks()
        self._streak = False

    def in_board(self):
        """Return current state of the board, reshaped to (1, BOARD_LEN, BOARD_LEN)
        for better symbiosis with torch.Tensor.
        """
        return np.array(self._board.copy(),dtype=bool).reshape((1, BOARD_LEN, BOARD_LEN))  # fmt: skip

    def in_blocks(self):
        """Return currently available block array(3,)."""
        tmp = self._available.copy()
        for i in range(len(tmp), 3):
            tmp.append(EMPTY_SLOT)
        return np.array(tmp, dtype=np.int8)

    def in_streak(self):
        """Return bool informing about streak, reshaped to array(1,)."""
        return np.array([self._streak], dtype=bool)

    @staticmethod
    def decode_action(action: int):
        """ Unpack flat encoded action index.

        Action is represented as tuple (piece_chosen, x_cord, y_cord), where: \\
        piece_chosen: int -> in range [1, BLOCKS_TO_PICK] \\
        x_cord: int -> in range [1, BOARD_LEN] \\
        y_cord: int -> in range [1, BOARD_LEN]
        """
        piece = action // BOARD_FLAT
        pos = action % BOARD_FLAT
        x = pos // BOARD_LEN
        y = pos % BOARD_LEN
        return piece, x, y

    @staticmethod
    def _get_valid_mask(board: np.ndarray, idx: int) -> np.ndarray:
        """Return mask of valid positions for placing BLOCKS[idx]."""
        block = BLOCKS[idx]
        valid_mask = np.ones(BOARD_SIZE, dtype=bool)
        n = BOARD_LEN

        for dx, dy in block:
            # fmt: off
            invalid_positions = board[max(0, dx):min(n, n + dx), 
                                    max(0, dy):min(n, n + dy)]
            # fmt: on

            current_valid = np.zeros(BOARD_SIZE, dtype=bool)
            # fmt: off
            current_valid[max(0, -dx):min(n, n - dx), 
                        max(0, -dy):min(n, n - dy)] = ~invalid_positions
            # fmt: on
            valid_mask &= current_valid

        return valid_mask

    def _calculate_strikes(self, block: list, x: int, y: int):
        """Calculate clearings of the board, after swipe moves.

        Modify inside state accordingly to those clearings.

        Args:
            block, x, y: Represent placement of a new block on a board.
            (Passed in for optimalization purposes)

        Returns:
            A pair (int, bool) informing on reward in points,
            and whether any clearing was made.
        """
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
            if self._board[sq_x : sq_x + 3, sq_y : sq_y + 3].all():
                hits.add((2, sq_x + sq_y * 8))
                hit = True

            if not hit:
                rwrd += 1
            else:
                streak = True
        # Execute clearings.
        for what, id in hits:
            if what == 0:
                self._board[id, :] = False
            elif what == 1:
                self._board[:, id] = False
            else:
                org_x = id % 8
                org_y = id // 8
                self._board[org_x : org_x + 3, org_y : org_y + 3] = False
        # Calculate reward.
        rwrd += streak * self._streak * State.STREAK_REWARD
        rwrd += len(hits) * State.CLEARING_REWARD
        return rwrd, streak

    def calculate_mask(self):
        """Calculate legal / illegal mask of all moves from current state.

        Value of mask[index] = 0 marks that move encoded as index is illegal.

        Returns:
            mask: Mask of legality of moves.
        """
        mask = np.zeros((BLOCKS_TO_PICK, BOARD_LEN, BOARD_LEN), bool)
        mask[0, :, :] = self._get_valid_mask(self._board, self._available[0])

        if self._count == 1:
            mask[1, :, :] = False
            mask[2, :, :] = False
        elif self._count == 2:
            mask[1, :, :] = self._get_valid_mask(self._board, self._available[1])
            mask[2, :, :] = False
        elif self._count == 3:
            mask[1, :, :] = self._get_valid_mask(self._board, self._available[1])
            mask[2, :, :] = self._get_valid_mask(self._board, self._available[2])
        else:
            raise ValueError(f"_count value: {self._count} is not in range 1-3!")
        return mask.reshape(NN_OUTPUT_FLAT)

    def transition_key(self, piece: int, x: int, y: int):
        """Transition state according to given action (piece, x, y).

        Transition occures in place and reward value of that action is returned.

        Returns:
            rwrd: Value of reward from transition T(state, action).
        """
        block_id = self._available[piece]
        block = BLOCKS[block_id]

        # make move
        self._count -= 1
        self._available.pop(piece)
        for dx, dy in BLOCKS[block_id]:
            self._board[x + dx, y + dy] = True

        # calculate reward and a new board
        rwrd, self._streak = self._calculate_strikes(block, x, y)
        if not self._board.any():
            rwrd += State.TOTAL_CLEANUP_REWARD

        # check if we need to sample new blocks
        if self._count == 0:
            self._available = rng_fact_instance.new_blocks()
            self._count = BLOCKS_TO_PICK

        return rwrd

    def transition(self, piece: int, x: int, y: int):
        """Alike transition_key, but return mask for state'.

        More computationally heavy.

        Returns:
            List holding three values -
            rwrd: Value of reward from transition T(state, action).
            terminal: Whether state' is terminal.
            mask: np.array masking legal moves from state'.
        """
        rwrd = self.transition_key(piece, x, y)
        mask = self.calculate_mask()
        if not mask.any():
            rwrd += NEURAL_PENALTY
        return [rwrd, bool(not mask.any()), mask]

    def all_1move_states(self, mask: np.ndarray):
        """Calculate all variants of moves from this state and thier afterstate.

        Returned numpy arrays are reshaped to work well as NN input, after being
        transformed to torch.Tensor.

        Args:
            mask: Mask of legal moves, returned from self.calculate_mask().

        Returns:
            Tuple (A, R, T, B, L, S), where:
            A: Valid actions, encoded as integers.
            R[i]: - reward from picking action A[i].
            T[i]: - is A[i] terminal.
            B[i]: - state of board after taking action A[i].
            L[i]: - blocks available after taking action A[i].
            S[i]: - is streak on after action A[i].
        """
        actions = np.flatnonzero(mask).astype(int)
        n_moves = len(actions)

        rewards = np.empty(n_moves, dtype=np.int16)
        terminals = np.empty(n_moves, dtype=bool)
        out_boards = np.empty((n_moves, 1, BOARD_LEN, BOARD_LEN), dtype=bool)
        out_blocks = np.empty((n_moves, BLOCKS_TO_PICK), dtype=np.int8)
        out_streaks = np.empty((n_moves, 1), dtype=bool)

        for i, a in enumerate(actions):
            piece, x, y = State.decode_action(a)
            new_state = self.copy()
            rewards[i], terminals[i], _ = new_state.transition(piece, x, y)
            out_boards[i] = new_state.in_board()
            out_blocks[i] = new_state.in_blocks()
            out_streaks[i] = new_state.in_streak()

        return actions, rewards, terminals, out_boards, out_blocks, out_streaks

    def copy(self):
        """Get a copy of the State object."""
        new_st = object.__new__(self.__class__)
        new_st._board = self._board.copy()
        new_st._count = self._count
        new_st._available = self._available.copy()
        new_st._streak = self._streak
        return new_st


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
    print(st.in_board())
    print(st.in_blocks())

    print("\n\nREWARD, IS_MOVE_FINISHING, MASK_OF_ILLEGAL_MOVES!")
    print(st.transition(0, 4, 4))

    print("\n\nAFTERWARDS STATE!")
    print(st._board.astype(int))

    print("\n\nAFTERWARDS NEURAL OUTPUT")
    print(st.in_board())
    print(st.in_blocks())

    print(type(st._available))
