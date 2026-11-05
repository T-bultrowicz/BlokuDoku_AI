import numpy as np
import random as rand
import torch
import os

""" Constants around Bloku Doku game logic """

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


# Handling different devices logic.
def _pick_device() -> torch.device:
    """Pick device for NNs: CUDA, then Apple MPS, then CPU.

    Can be forced with environment variable BLOKUDOKU_DEVICE (e.g. "cpu", "cuda:1").
    """
    forced = os.environ.get("BLOKUDOKU_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _pick_device()
if DEVICE.type == "cuda":
    # Input shapes are fixed, let cuDNN pick fastest kernels.
    torch.backends.cudnn.benchmark = True

# Anything to_tensor() accepts.
TensorLike = np.ndarray | torch.Tensor


def to_tensor(
    arr: TensorLike, dtype: torch.dtype = torch.float32, requires_grad: bool = False
) -> torch.Tensor:
    """Convert np.array or tensor to tensor of given dtype, placed on DEVICE,
    sets required_grad as given (default=False). No copy is made when arr
    already matches DEVICE and dtype."""
    # fmt: off
    return torch.as_tensor(arr, device=DEVICE) \
                .to(dtype) \
                .requires_grad_(requires_grad)
    # fmt: on


# Randomness parameter
DEF_SEED = 13579


# Factory used to generate states in BlokuDoku training games.
class RandomFactory:
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


rng_fact_instance = RandomFactory(DEF_SEED)


def _calc_constant_tables():
    """Calculate constant tables computed once, at the start of training process.

    Returns:
        PLACEMENTS: np.array of shape (BLOCK_SIZE + 1, BOARD_FLAT, BOARD_FLAT), where
        P[block, pos] = mask of all positions that block covers.
        IN_BOUNDS: np.array of shape (BLOCK_SIZE + 1, BOARD_FLAT), where
        I[block] = mask of all positions where block can be placed.
        LINES: (3 * BOARD_LEN, BOARD_FLAT), rows/columns/squares masks.
    """
    pl = np.zeros((BLOCKS_SIZE + 1, BOARD_FLAT, BOARD_FLAT), dtype=bool)
    ib = np.zeros((BLOCKS_SIZE + 1, BOARD_FLAT), dtype=bool)
    lines = np.zeros((3 * BOARD_LEN, BOARD_FLAT), dtype=bool)
    # Placements and in bounds masks generating
    # ib[EMPTY_SLOT] stays all False - as it cannot be placed.
    for i, block in enumerate(BLOCKS):
        block = np.array(block)
        for j in range(BOARD_FLAT):
            xs = j // BOARD_LEN + block[:, 0]
            ys = j % BOARD_LEN + block[:, 1]
            if ((xs >= 0) & (xs < BOARD_LEN) & (ys >= 0) & (ys < BOARD_LEN)).all():
                pl[i, j, xs * BOARD_LEN + ys] = True
                ib[i, j] = True

    # Lines masks generating
    columns = np.array([BOARD_LEN * i for i in range(BOARD_LEN)])
    for i in range(BOARD_LEN):
        lines[i, (columns + i)] = True
    rows = np.arange(BOARD_LEN, dtype=np.int32)
    for i in range(BOARD_LEN):
        lines[i + BOARD_LEN, rows + (i * BOARD_LEN)] = True
    help = np.array([0, 1, 2])
    squares = np.concat([help, BOARD_LEN + help, 2 * BOARD_LEN + help]) - 21
    for i in range(BOARD_LEN):
        squares += (0 if i % 3 else 2 * BOARD_LEN) + 3
        lines[i + 18, squares] = True

    return (
        to_tensor(pl, torch.bool),
        to_tensor(ib, torch.bool),
        to_tensor(lines, torch.bool),
    )


class VectorizedStates:
    """Class capable of efficiently storing and transforming many states of
    BlokuDoku game at once.

    This class by default will be constructed on the same device where NN
    will be trained, along with constast tables required for instances
    of this class to work more efficient.
    """

    MAX_BOARDS = 10_000
    MM_DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32
    CHUNK = 8192

    # SHAPES: (48, 81, 81), (48, 81), (27, 81). (47 + EMPTY_BLOCK)
    PLACEMENTS, IN_BOUNDS, LINES = _calc_constant_tables()
    PLACEMENTS_FLAT = PLACEMENTS.view(-1, BOARD_FLAT)
    PLACEMENTS_MM = PLACEMENTS_FLAT.to(MM_DTYPE)
    PLACEMENTS_T = PLACEMENTS_MM.T.contiguous()
    LINES_T = LINES.to(MM_DTYPE).T.contiguous()  # (81, 27)
    OFFSETS = torch.arange(BOARD_FLAT, device=DEVICE)

    TOTAL_CLEAR_RWRD = 100
    CLEAR_RWRD = 18
    STREAK_RWRD = 9

    def __init__(self, n: int = 1):
        """Creates VectorizedStates instance.

        Args:
            n: Number of states to be stored inside vector.
        """
        if n <= 0:
            raise ValueError("Number of boards cannot be negative!")
        if n > self.MAX_BOARDS:
            raise ValueError("Number of boards to create in parallel is to big!")

        self.N = n
        self.boards = to_tensor(np.zeros((n, BOARD_FLAT), bool), torch.bool)
        self.available = self._gen_available()
        self.streaks = torch.zeros((n, 1), dtype=torch.bool, device=DEVICE)

    def _move_indices(self):
        """Returns indices for PLACEMENTS cols that need to be checked."""
        # long() avoids overflow of uint8.
        base = self.available.long().unsqueeze(-1) * BOARD_FLAT
        return (base + self.OFFSETS).flatten(start_dim=1)

    def _gen_available(self):
        """Randomise new available blocks."""
        return torch.randint(
            0, BLOCKS_SIZE, (self.N, BLOCKS_TO_PICK), device=DEVICE, dtype=torch.uint8
        )

    def legal_moves(self):
        """Calculate mask of legal moves.

        Returns:
            mask: of (N, BLOCKS_TO_PICK * BOARD_FLAT) bools, where mask[gid, id]
            informs whether move encoded as id is legal to play on game #gid.
        """
        indices = self._move_indices()
        # (N, 81) @ (81, (BLOCKS_SIZE + 1) * 81) -> overlap for every block.
        overlap = self.boards.to(self.MM_DTYPE) @ self.PLACEMENTS_T
        overlap = torch.gather(overlap, dim=1, index=indices)
        # Pick only columns of available blocks, that are in bounds.
        return (overlap == 0) & self.IN_BOUNDS.view(-1)[indices]

    def legal_moves2(self):
        """Same as legal_moves, but instead of one big matrix multiply,
        gathers placement masks of available blocks only and performs
        N small multiplies (1, 81) @ (81, BLOCKS_TO_PICK * 81) via bmm.
        """
        indices = self._move_indices()
        # (N, 243, 81) -> transposed view (N, 81, 243), bmm handles strides.
        masks = self.PLACEMENTS_MM[indices].transpose(1, 2)
        overlap = torch.bmm(self.boards.to(self.MM_DTYPE).unsqueeze(1), masks)
        return (overlap.squeeze(1) == 0) & self.IN_BOUNDS.view(-1)[indices]

    def _simulate_transitions(
        self, mask: torch.Tensor, boards: torch.Tensor, fills: torch.Tensor
    ):
        """ Calculate all logic of making correct moves from mask. Obtain rewards, 
        terminals and new boards - only for legal moves.

        Args:
            boards: (M, BOARD_FLAT) bools, boards right after placing a block.
            placed: (M, BOARD_FLAT) bools, cells covered by the placed block.
            old_streaks: (M,) bools, streak flags before the move.

        Returns:
            Tuple (rewards, boards, streaks) of shapes (M,), (M, BOARD_FLAT), (M,):
            reward without NEURAL_PENALTY, board after clearings, new streak flag.
        """
        B = boards[mask]
        F = fills[mask]
        S_old = self.streaks.view(-1, 1).expand(mask.shape)[mask]

        counts = B.to(self.MM_DTYPE) @ self.LINES_T  # (M, 81) @ (81, 27) = (M, 27)
        hits = counts == BOARD_LEN  #  Only columns were sum = BOARD_LEN
        cleared = (hits.to(self.MM_DTYPE) @ self.LINES.to(self.MM_DTYPE)) > 0
        streaks = hits.any(dim=-1)
        B = B & ~cleared

        rewards = (F & ~cleared).sum(dim=-1)
        rewards += hits.sum(dim=-1) * self.CLEAR_RWRD
        rewards += (streaks & S_old) * self.STREAK_RWRD
        rewards += ~B.any(dim=-1) * self.TOTAL_CLEAR_RWRD
        return rewards, B, streaks

    def _has_legal_move(self, boards: torch.Tensor, blocks: torch.Tensor):
        """Whether any legal move exists, for M arbitrary states.

        Args:
            boards: (M, BOARD_FLAT) bools.
            blocks: (M, BLOCKS_TO_PICK) block ids, EMPTY_SLOT allowed.

        Returns:
            (M,) bools.
        """
        out = torch.empty(len(boards), dtype=torch.bool, device=DEVICE)
        # Full overlap table is (chunk, 49 * 81) - hence chunking.
        for s in range(0, len(boards), self.CHUNK):
            overlap = boards[s : s + self.CHUNK].to(self.MM_DTYPE) @ self.PLACEMENTS_T
            overlap = overlap.view(-1, BLOCKS_SIZE + 1, BOARD_FLAT)
            fits = ((overlap == 0) & self.IN_BOUNDS).any(dim=-1)  # (chunk, 49)
            out[s : s + self.CHUNK] = fits.gather(
                1, blocks[s : s + self.CHUNK].long()
            ).any(dim=-1)
        return out

    def states_prim(self):
        """Calculate all variants of moves from states to their afterstates.

        Keeps current state unchanged, every element of returned tuple is
        torch.Tensor constructed on proper device. Entries of illegal actions
        are meaningless (zeros / board with block placed on top).

        Returns:
            Tuple (A, R, T, B, L, S), where:
            A[i]: - Is action encoded as i legal. (N, 243)
            R[i]: - reward from picking action i. (N, 243)
            T[i]: - is action i terminal. (N, 243)
            B[i]: - state of board after taking action i. (N, 243, 81)
            L[i]: - blocks available after taking action i. (N, 243, 3)
            S[i]: - is streak on after action i. (N, 243)
        """
        N = self.N
        A = self.legal_moves()  # (N, 243)
        placed = self.PLACEMENTS_FLAT[self._move_indices()]  # (N, 243, 81)
        B = self.boards[:, None, :] | placed

        # Slot k is used up by actions [k * 81, (k + 1) * 81).
        L = self.available[:, None, :].repeat(1, NN_OUTPUT_FLAT, 1)
        L = L.view(N, BLOCKS_TO_PICK, BOARD_FLAT, BLOCKS_TO_PICK)
        for k in range(BLOCKS_TO_PICK):
            L[:, k, :, k] = EMPTY_SLOT
        L = L.view(N, NN_OUTPUT_FLAT, BLOCKS_TO_PICK)

        empty = (L == EMPTY_SLOT).all(dim=-1, keepdim=True)
        fresh = self._gen_available()[:, None, :]  # (N, 1, 3)
        L = torch.where(empty, fresh, L)

        # Clearings, rewards and terminals only for legal moves.
        rewards, boards, streaks = self._simulate_transitions(A, B, placed)
        terminals = ~self._has_legal_move(boards, L[A])
        rewards += terminals * NEURAL_PENALTY

        R = torch.zeros((N, NN_OUTPUT_FLAT), dtype=torch.int16, device=DEVICE)
        T = torch.zeros((N, NN_OUTPUT_FLAT), dtype=torch.bool, device=DEVICE)
        S = torch.zeros((N, NN_OUTPUT_FLAT), dtype=torch.bool, device=DEVICE)
        B[A] = boards
        R[A] = rewards.to(torch.int16)
        T[A] = terminals
        S[A] = streaks
        return A, R, T, B, L, S

    def mass_transition(
        self, boards: torch.Tensor, available: torch.Tensor, streaks: torch.Tensor
    ):
        self.boards = boards
        self.available = available
        self.streaks = streaks

    def reset(self, mask: torch.Tensor):
        self.boards[mask] = False
        self.available = torch.where(mask[:, torch.newaxis], self._gen_available(), self.available)
        self.streaks[mask] = False

    def nn_input(self):
        B = self.boards.reshape(self.N, 1, BOARD_LEN, BOARD_LEN)
        S = self.streaks.reshape(self.N, 1)
        return B, self.available, S


class State:
    """Backward compatibility wrapper: single game as VectorizedStates(1),
    numpy in/out. Kept for sequential code paths (showcases, user play).
    """

    def __init__(self):
        self._vs = VectorizedStates(1)

    def in_board(self):
        return self._vs.boards.cpu().numpy().reshape((1, BOARD_LEN, BOARD_LEN))

    def in_blocks(self):
        return self._vs.available[0].cpu().numpy().astype(np.int8)

    def in_streak(self):
        return self._vs.streaks.cpu().numpy().reshape(1)

    @staticmethod
    def decode_action(action: int):
        """Flat action index -> (piece, x, y)."""
        piece = action // BOARD_FLAT
        pos = action % BOARD_FLAT
        return piece, pos // BOARD_LEN, pos % BOARD_LEN

    @staticmethod
    def encode_action(piece: int, x: int, y: int) -> int:
        return piece * BOARD_FLAT + x * BOARD_LEN + y

    def calculate_mask(self):
        return self._vs.legal_moves()[0].cpu().numpy()

    def transition(self, piece: int, x: int, y: int):
        """In place move, returns [reward, terminal, mask of state']."""
        a = State.encode_action(piece, x, y)
        A, R, T, B, L, S = self._vs.states_prim()
        if not A[0, a]:
            raise ValueError(f"Illegal action: piece={piece}, x={x}, y={y}")
        self._vs.mass_transition(B[0, a : a + 1], L[0, a : a + 1], S[0, a : a + 1].view(1, 1))
        return [int(R[0, a]), bool(T[0, a]), self.calculate_mask()]

    def all_1move_states(self, mask: np.ndarray | None = None):
        """Legal actions and their afterstates as numpy (A, R, T, B, L, S).
        mask is ignored, kept for the old signature."""
        A, R, T, B, L, S = (t[0] for t in self._vs.states_prim())
        actions = A.nonzero().flatten()
        return (
            actions.cpu().numpy(),
            R[actions].cpu().numpy(),
            T[actions].cpu().numpy(),
            B[actions].cpu().numpy().reshape(-1, 1, BOARD_LEN, BOARD_LEN),
            L[actions].cpu().numpy().astype(np.int8),
            S[actions].cpu().numpy().reshape(-1, 1),
        )
