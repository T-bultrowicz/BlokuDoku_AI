import blokudoku_lib as bl
import user_play as user_play

import numpy as np
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

GAMES_TO_PLAY = 3
GAMES_TO_EXCHANGE_NETS = 40
GAMES_TO_DISPLAY_INFO = 1000
EPS_MAX = 1.0
EPS_MIN = 0.025
DECAY_RATE = 0.999995
DISCOUNT = 0.99
LEARN_MOVE = 4
BATCH_SIZE = 32


IN_CONV_CHANNELS = 1
OUT_CONV_CHANNELS_1 = 32
OUT_CONV_CHANNELS_2 = 64
OUT_CONV_CHANNELS_3 = 64
OUT_CONV_CHANNELS_4 = 16
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
class RecordStorage():
    def __init__(self):
        self.boards_s = np.zeros((RECORDS_CAPACITY, 1, bl.BOARD_LEN, bl.BOARD_LEN), dtype=bool)
        self.blocks_s = np.zeros((RECORDS_CAPACITY, bl.BLOCKS_TO_PICK), dtype=np.int8)
        self.streaks_s = np.zeros((RECORDS_CAPACITY, 1), dtype=bool)

        self.boards_f = np.zeros((RECORDS_CAPACITY, 1, bl.BOARD_LEN, bl.BOARD_LEN), dtype=bool)
        self.blocks_f = np.zeros((RECORDS_CAPACITY, bl.BLOCKS_TO_PICK), dtype=np.int8)
        self.streaks_f = np.zeros((RECORDS_CAPACITY, 1), dtype=bool)

        self.actions = np.zeros(RECORDS_CAPACITY, dtype=np.int16)
        self.rewards = np.zeros(RECORDS_CAPACITY, dtype=np.uint16)
        self.terminal = np.zeros(RECORDS_CAPACITY, dtype=bool)
        self.masks_f = np.zeros((RECORDS_CAPACITY, OUT_LINEAR_DIM), dtype=bool)

        self.offset = 0
        self.size = 0

    def add_record(self, board_s: np.ndarray, blocks_s: np.ndarray, streak_s: bool,
                         board_f: np.ndarray, blocks_f: np.ndarray, streak_f: bool,
                         action: int, reward: int, terminal: bool, mask_f: np.ndarray):
        self.boards_s[self.offset] = [board_s]
        self.blocks_s[self.offset] = blocks_s
        self.streaks_s[self.offset] = [streak_s]

        self.boards_f[self.offset] = [board_f]
        self.blocks_f[self.offset] = blocks_f
        self.streaks_f[self.offset] = [streak_f]

        self.actions[self.offset] = action
        self.rewards[self.offset] = reward
        self.terminal[self.offset] = terminal
        self.masks_f[self.offset] = mask_f

        self.offset = (self.offset + 1) % RECORDS_CAPACITY
        self.size = min(self.size + 1, RECORDS_CAPACITY)

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

def train():
    main_net = BLOCK_Q_Network()
    target_net = BLOCK_Q_Network()
    storage = RecordStorage()

    moves = 0
    eps = EPS_MAX
    batch_start_time = time.perf_counter()
    optimizer = torch.optim.Adam(main_net.parameters(), lr=1e-4)
    all_rewards = []

    for game_num in range(GAMES_TO_PLAY):
        st_nw = bl.State()
        curr_mask = st_nw.calculate_mask()
        game_over = False
        total_rwrd = 0

        while not game_over:
            eps = max(EPS_MIN, eps * DECAY_RATE)
            board, blocks, streak = st_nw.in_board(), st_nw.in_blocks(), st_nw.in_streak()

            nn_board = torch.from_numpy(board).float().unsqueeze(0).unsqueeze(0)
            nn_blocks = torch.from_numpy(blocks).long().unsqueeze(0)
            nn_streak = torch.from_numpy(np.array([[streak]], dtype=np.float32))

            with torch.no_grad():
                y_hat_nn = main_net(nn_board, nn_blocks, nn_streak).squeeze(0)
                y_hat_nn[~curr_mask] = -1e9

            action = 0
            if bl.rng_fact_instance.explore_now(eps):
                actionss = np.flatnonzero(curr_mask).tolist()
                action = bl.rng_fact_instance.choice(actionss)
            else:
                action = int(torch.argmax(y_hat_nn))
            piece, x, y = bl.decode_action(action)
            
            rwrd, is_terminal, new_mask = st_nw.transition(piece, x, y)
            storage.add_record(
                board, blocks, streak,
                st_nw.in_board(), st_nw.in_blocks(), st_nw.in_streak(),
                action, rwrd, is_terminal, new_mask
            )

            # PERFORM LEARNING STEP ON THE MAIN NET
            if moves % LEARN_MOVE == 0 and storage.size >= BATCH_SIZE:
                (in_brd_st1, in_blk_st1, in_str_st1, 
                 in_brd_st2, in_blk_st2, in_str_st2, 
                 actions, rewards, terminals, masks_f) = storage.sample(BATCH_SIZE)
                terminals = torch.from_numpy(terminals).float()
                rewards = torch.from_numpy(rewards).float()
                
                y_hat_nn_main = main_net(
                    torch.from_numpy(in_brd_st1).float(),
                    torch.from_numpy(in_blk_st1).long(),
                    torch.from_numpy(in_str_st1).float()
                )
                actions_t = torch.from_numpy(actions).long()
                q_pred_fn = y_hat_nn_main[torch.arange(BATCH_SIZE), actions_t]

                with torch.no_grad():
                    y_hat_nn_target = target_net(
                        torch.from_numpy(in_brd_st2).float(),
                        torch.from_numpy(in_blk_st2).long(),
                        torch.from_numpy(in_str_st2).float()
                    )
                    y_hat_nn_target[~masks_f] = -1e9
                    target_moves = torch.max(y_hat_nn_target, dim=1).values
                
                target_fn = rewards + (1 - terminals) * DISCOUNT * target_moves
                losses = F.mse_loss(q_pred_fn, target_fn)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

            game_over = is_terminal
            moves += 1
            curr_mask = new_mask
            total_rwrd += rwrd

        all_rewards.append(total_rwrd)
        games_played = game_num + 1
        if games_played % GAMES_TO_DISPLAY_INFO == 0:
            # PRINT STATISTICS OF TRAINING
            elapsed = time.perf_counter() - batch_start_time
            games_per_second = GAMES_TO_DISPLAY_INFO / elapsed if elapsed else float("inf")

            print(f"Played {games_played} games, last batch of {GAMES_TO_DISPLAY_INFO} games:")
            print(f"Average points per game: {np.mean(all_rewards)}, eps: {eps}!")
            print(f"Batch time: {elapsed:.2f}s, games per second: {games_per_second:.2f}\n\n")

            batch_start_time = time.perf_counter()
            all_rewards.clear()

        if games_played % GAMES_TO_EXCHANGE_NETS == 0:
            # LOAD TARGET NET WITH MAIN NET'S WEIGHTS
            target_net.load_state_dict(main_net.state_dict())

    main_net_path = "main_net.pt"
    torch.save(main_net.state_dict(), main_net_path)

            
            
                
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
