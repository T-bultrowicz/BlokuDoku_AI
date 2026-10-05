import blokudoku_lib as bl
import user_play as user_play

import numpy as np
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

MAIN_NET_PATH = "main_net.pt"

GAMES_TO_PLAY = 125000
GAMES_IN_PARALLEL = 16

EXCHANGE_NETS_FREQ = 250
GAMES_TO_DISPLAY_INFO = 1000

VALIDATION_GAMES_PER_BATCH = 5
EPS_MAX = 1.0
EPS_MIN = 0.025
DECAY_RATE = 0.999995
DISCOUNT = 0.999
BATCH_SIZE = 128
LEARING_OPTIM_RATE = 1e-4

IN_CONV_CHANNELS = 1
OUT_CONV_CHANNELS_1 = 32
OUT_CONV_CHANNELS_2 = 64
OUT_CONV_CHANNELS_3 = 64
OUT_CONV_CHANNELS_4 = 32
KERNEL_SIZE = (3, 3)
PADDING = 1

OUT_EMBEDDING_DIM = 16

IN_LINEAR_DIM = (OUT_CONV_CHANNELS_4 * bl.BOARD_FLAT) + (OUT_EMBEDDING_DIM * bl.BLOCKS_TO_PICK) + 1
HID_LINEAR_DIM = 256
OUT_LINEAR_DIM = bl.BLOCKS_TO_PICK * bl.BOARD_FLAT

class BLOCK_Q_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(IN_CONV_CHANNELS, OUT_CONV_CHANNELS_1, KERNEL_SIZE, padding=PADDING),
            nn.ReLU(),
            nn.Conv2d(OUT_CONV_CHANNELS_1, OUT_CONV_CHANNELS_2, KERNEL_SIZE, padding=PADDING),
            nn.ReLU(),
            nn.Conv2d(OUT_CONV_CHANNELS_2, OUT_CONV_CHANNELS_3, KERNEL_SIZE, padding=PADDING),
            nn.ReLU(),
            nn.Conv2d(OUT_CONV_CHANNELS_3, OUT_CONV_CHANNELS_4, KERNEL_SIZE, padding=PADDING),
            nn.ReLU(),
            nn.Flatten()
        )

        self.embed_net = nn.Sequential(
            nn.Embedding(bl.BLOCKS_SIZE + 1, OUT_EMBEDDING_DIM),
            nn.Flatten()
        )

        self.afterwards_net = nn.Sequential(
            nn.Linear(IN_LINEAR_DIM, HID_LINEAR_DIM),
            nn.ReLU(),
            nn.Linear(HID_LINEAR_DIM, HID_LINEAR_DIM),
            nn.ReLU(),
            nn.Linear(HID_LINEAR_DIM, OUT_LINEAR_DIM),
        )

    def forward(self, board: torch.Tensor, blocks: torch.Tensor, streak: torch.Tensor):
        board_out = self.conv_net(board)
        blocks_out = self.embed_net(blocks)

        final_input = torch.cat((board_out, blocks_out, streak), dim=1)
        return self.afterwards_net(final_input)

RECORDS_CAPACITY = 100000

class RecordStorage:
    def __init__(self):
        self.boards_s = np.empty((RECORDS_CAPACITY, 1, bl.BOARD_LEN, bl.BOARD_LEN), dtype=bool)
        self.blocks_s = np.empty((RECORDS_CAPACITY, bl.BLOCKS_TO_PICK), dtype=np.int8)
        self.streaks_s = np.empty((RECORDS_CAPACITY, 1), dtype=bool)

        self.boards_f = np.empty((RECORDS_CAPACITY, 1, bl.BOARD_LEN, bl.BOARD_LEN), dtype=bool)
        self.blocks_f = np.empty((RECORDS_CAPACITY, bl.BLOCKS_TO_PICK), dtype=np.int8)
        self.streaks_f = np.empty((RECORDS_CAPACITY, 1), dtype=bool)

        self.actions = np.empty(RECORDS_CAPACITY, dtype=np.int16)
        self.rewards = np.empty(RECORDS_CAPACITY, dtype=np.int16)
        self.terminal = np.empty(RECORDS_CAPACITY, dtype=bool)
        self.masks_f = np.empty((RECORDS_CAPACITY, OUT_LINEAR_DIM), dtype=bool)

        self.offset = 0
        self.size = 0

    def _write_arrays(self, incoming_data: tuple, n: int):
        buffers = (
            self.boards_s, self.blocks_s, self.streaks_s,
            self.boards_f, self.blocks_f, self.streaks_f,
            self.actions, self.rewards, self.terminal, self.masks_f
        )

        if self.offset + n <= RECORDS_CAPACITY:
            for buf, data in zip(buffers, incoming_data):
                buf[self.offset : self.offset + n] = data
        else:
            space_left = RECORDS_CAPACITY - self.offset
            remainder = n - space_left

            for buf, data in zip(buffers, incoming_data):
                buf[self.offset : RECORDS_CAPACITY] = data[:space_left]
                buf[0 : remainder] = data[space_left:]

    def add_records(self, boards_s: np.ndarray, blocks_s: np.ndarray, streaks_s: np.ndarray,
                          boards_f: np.ndarray, blocks_f: np.ndarray, streaks_f: np.ndarray,
                          actions: np.ndarray, rewards: np.ndarray, terminals: np.ndarray, 
                          masks_f: np.ndarray):
        
        n = len(boards_s) # Amount of records to add.
        incoming_data = (
            boards_s, blocks_s, streaks_s,
            boards_f, blocks_f, streaks_f,
            actions, rewards, terminals, masks_f
        )

        self._write_arrays(incoming_data, n)
        self.offset = (self.offset + n) % RECORDS_CAPACITY
        self.size = min(self.size + n, RECORDS_CAPACITY)

    def sample(self, num_samples: int = 1):
        idx = bl.rng_fact_instance.sample(range(self.size), num_samples)
        return (
            self.boards_s[idx],
            self.blocks_s[idx],
            self.streaks_s[idx],
            self.boards_f[idx],
            self.blocks_f[idx],
            self.streaks_f[idx],
            self.actions[idx],
            self.rewards[idx],
            self.terminal[idx],
            self.masks_f[idx]
        )

def validate_model(model: BLOCK_Q_Network, num_games: int = 1):
    total_score = 0
    for _ in range(num_games):
        st_nw = bl.State()
        curr_mask = st_nw.calculate_mask()
        game_over = False
        while not game_over:
            board, blocks, streak = st_nw.in_board(), st_nw.in_blocks(), st_nw.in_streak()

            nn_board = torch.from_numpy(board).float().unsqueeze(0)
            nn_blocks = torch.from_numpy(blocks).long().unsqueeze(0)
            nn_streak = torch.from_numpy(streak).float().view(1, 1)

            with torch.no_grad():
                y_hat_nn = model(nn_board, nn_blocks, nn_streak).squeeze(0)
                y_hat_nn[~curr_mask] = -1e9

            action = int(torch.argmax(y_hat_nn))
            piece, x, y = bl.decode_action(action)

            rwrd, is_terminal, new_mask = st_nw.transition(piece, x, y)
            game_over = is_terminal
            curr_mask = new_mask
            total_score += rwrd

    return f"Average points per game: {(total_score / num_games):.4f}"

def _randomise(mask: np.ndarray):
    valid_actions = np.flatnonzero(mask)
    return bl.rng_fact_instance.choice(valid_actions)

def train(games_count: int = GAMES_IN_PARALLEL):
    main_net = BLOCK_Q_Network()
    target_net = BLOCK_Q_Network()
    storage = RecordStorage()

    optimizer = torch.optim.Adam(main_net.parameters(), lr=LEARING_OPTIM_RATE)
    batch_start_time = time.perf_counter()
    eps = EPS_MAX

    states = [bl.State() for _ in range(games_count)]
    games_played = 0
    games_played_mod = 0
    loops = 0
    while games_played < GAMES_TO_PLAY:
        masks = np.array([st.calculate_mask() for st in states])
        boards = [st.in_board() for st in states]
        blocks = [st.in_blocks() for st in states]
        streaks = [st.in_streak() for st in states]

        nn_boards = torch.from_numpy(np.array(boards)).float()
        nn_blocks = torch.from_numpy(np.array(blocks)).long()
        nn_streaks = torch.from_numpy(np.array(streaks)).float()

        with torch.no_grad():
            y_hat_nn = main_net(nn_boards, nn_blocks, nn_streaks)
            y_hat_nn[~torch.from_numpy(masks)] = -1e9
            best_actions = torch.argmax(y_hat_nn, dim=1).tolist()

        decisions = [bl.rng_fact_instance.explore_now(eps)  for _ in range(games_count)]
        actions = [
            (_randomise(masks[i]) if decisions[i] else best_actions[i]) 
            for i in range(games_count)
        ]
        decoded_actions = [bl.decode_action(actions[i]) for i in range(games_count)]

        rewards = np.empty(games_count, dtype=np.int16)
        terminals = np.empty(games_count, dtype=bool)
        new_masks = np.empty((games_count, OUT_LINEAR_DIM), dtype=bool)
        for i in range(games_count):
            rewards[i], terminals[i], new_masks[i] = states[i].transition(*decoded_actions[i])

        storage.add_records(
            np.array(boards), np.array(blocks), np.array(streaks),
            np.array([st.in_board() for st in states]),
            np.array([st.in_blocks() for st in states]),
            np.array([st.in_streak() for st in states]),
            np.array(actions), rewards, terminals, new_masks
        )

        if storage.size >= BATCH_SIZE:
            (in_brd_st1, in_blk_st1, in_str_st1, 
             in_brd_st2, in_blk_st2, in_str_st2, 
             b_actions, b_rewards, b_terminals, b_masks_f) = storage.sample(BATCH_SIZE)
            b_terminals = torch.from_numpy(b_terminals).float()
            b_rewards = torch.from_numpy(b_rewards).float()
            
            y_hat_nn_main = main_net(
                torch.from_numpy(in_brd_st1).float(),
                torch.from_numpy(in_blk_st1).long(),
                torch.from_numpy(in_str_st1).float()
            )
            actions_t = torch.from_numpy(b_actions).long()
            q_pred_fn = y_hat_nn_main[torch.arange(BATCH_SIZE), actions_t]

            with torch.no_grad():
                y_hat_nn_target = target_net(
                    torch.from_numpy(in_brd_st2).float(),
                    torch.from_numpy(in_blk_st2).long(),
                    torch.from_numpy(in_str_st2).float()
                )
                y_hat_nn_target[~b_masks_f] = -1e9
                target_moves = torch.max(y_hat_nn_target, dim=1).values
            
            target_fn = b_rewards + (1 - b_terminals) * DISCOUNT * target_moves
            losses = F.mse_loss(q_pred_fn, target_fn)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        loops += 1
        eps = max(EPS_MIN, eps * (DECAY_RATE ** games_count))
        games_played += np.sum(terminals)
        games_played_mod += np.sum(terminals)
        for i in range(games_count):
            if terminals[i]:
                states[i] = bl.State()
    
        if loops % EXCHANGE_NETS_FREQ == 0:
            target_net.load_state_dict(main_net.state_dict())

        if games_played_mod >= GAMES_TO_DISPLAY_INFO:
            elapsed = time.perf_counter() - batch_start_time
            games_per_second = GAMES_TO_DISPLAY_INFO / elapsed if elapsed else float("inf")

            print(f"Games played: {games_played}, Games per second: {games_per_second:.2f}, batch time: {elapsed:.2f}s, eps: {eps:.4f}")
            print(f"Validating model performance...")
            print(validate_model(main_net, VALIDATION_GAMES_PER_BATCH))
            print("\n")

            batch_start_time = time.perf_counter()
            games_played_mod %= GAMES_TO_DISPLAY_INFO

    torch.save(main_net.state_dict(), MAIN_NET_PATH)         
                
def main(args):
    if len(args) != 2:
       print(f"Main needs one argument, but received: {len(args) - 1} arguments! Type './executable usage' for help!")
       return 0

    if args[1] == "train":
        train()
    elif args[1] == "test":
        raise RuntimeError("Not implemented yet!")
    elif args[1] == "play_as_player":
        user_play.play_as_player()
    elif args[1] == "watch_ai_play":
        raise RuntimeError("Not implemented yet!")
    elif args[1] == "usage":
        print(
    """Commands available for blokudoku_AI:
    train: trains the AI model
    test: tests the AI model
    play_as_player: play the game as a player, to compare score with the AI
    watch_ai_play: AI plays slowly, so you can watch it play
    """)
    else:
        print(f"Unrecgonised argument: {args[1]}! Use command 'usage' for help! ")
        return 0


if __name__ == "__main__":
    main(sys.argv)
