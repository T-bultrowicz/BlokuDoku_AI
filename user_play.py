import blokudoku_lib as bl
import numpy as np

DEFAULT_COLOUR = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
GRAY = "\033[90m"

def _prints(s: str):
    print(s, end="")

def _print_board(board: np.ndarray, is_block: bool = False):
    X, Y = board.shape
    on_colour = YELLOW if is_block else RED
    colours = [GREEN, on_colour, BLUE]
    sep = [" ", "  "]
    signs = [".", "X", "X"]

    for x in range(1, X + 1):
        _prints(f"{GRAY} {x}")
        if x % 3 == 0 and not is_block:
            _prints(" ")
    print()
    for y in range(Y):
        _prints(f"{GRAY}{y + 1}")
        for x in range(X):
            _prints(f"{colours[board[x, y]]}{signs[board[x, y]]}{sep[x > 8]}")
            if x % 3 == 2 and x < X - 1 and not is_block:
                _prints(f"{GRAY}|")
        print()
        if y % 3 == 2 and y < Y - 1 and not is_block:
            print(f"{GRAY} {'-' * (2 * X + (max(0, X - 9)) + (X // 3))}")
    print(f"{DEFAULT_COLOUR}")

def _draw_block(arr: np.ndarray, block: list, offset: int = 0):
    min_x, min_y = np.min(block, axis=0)

    fx, fy = block[0]
    arr[fx - min_x + offset, fy - min_y] = 2
    for dx, dy in block[1:]:
        arr[dx - min_x + offset, dy - min_y] = 1

def _print_blocks(blocks: np.ndarray):
    arr = np.zeros((18, 5), dtype=np.int8)
    offset = 0
    for block in blocks:
        if block == bl.EMPTY_SLOT:
            continue
        _draw_block(arr, bl.BLOCKS[block], offset)
        offset += 6
    _print_board(arr, is_block=True)

    

def _output_to_player(st: bl.State, score: int, terminal: bool):
    print(f"Score: {score}, streak: {"ON" if st.in_streak() else "OFF"}")
    if terminal:
        print("Game Over!")
    else:
        _print_board(st.in_board().astype(np.int8).squeeze(0))
        _print_blocks(st.in_blocks())

def _input_from_player(state: bl.State):
    prompt = "Give number of pawn, then lateral coordinate, finally vertical coordinate: "
    while True:
        try:
            values = list(map(int, input(prompt).split()))
        except ValueError:
            print("Invalid input. Give three whole numbers.")
            continue

        if len(values) != 3:
            print("Invalid input. Give three whole numbers.")
            continue

        piece, x, y = (value - 1 for value in values)
        blocks = state.in_blocks()
        if not 0 <= piece < len(blocks) or blocks[piece] == bl.EMPTY_SLOT:
            print("Invalid pawn number.")
            continue

        board = state.in_board().squeeze(0)
        block = bl.BLOCKS[int(blocks[piece])]
        positions = [(x + dx, y + dy) for dx, dy in block]
        if any(
            not (0 <= pos_x < board.shape[0] and 0 <= pos_y < board.shape[1])
            or board[pos_x, pos_y]
            for pos_x, pos_y in positions
        ):
            print("Invalid move.")
            continue

        return piece, x, y

def play_as_player():
    bl.rng_fact_instance = bl.RandomFactory()
    state = bl.State()
    game_over = False
    total_score = 0

    while not game_over:
        _output_to_player(state, total_score, game_over)
        piece, x, y = _input_from_player(state)
        rwrd, game_over, _ = state.transition(piece, x, y) 
        total_score += rwrd
    total_score -= bl.NEURAL_PENALTY
    _output_to_player(state, total_score, game_over)

if __name__ == "__main__":
    play_as_player()