import blokudoku_lib as bl

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, ker):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, 
                               kernel_size=ker, padding='same')
        self.conv2 = nn.Conv2d(in_channels=ch_out, out_channels=ch_out, 
                               kernel_size=ker, padding='same')

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(out + residual)

class BL_DQN_Conv_Network(nn.Module):
    IN_CON_CHANNELS = 1
    OUT_GEN_CONV = 64
    OUT_SUM_CONV = 32
    
    KER_LAT = (1, 9)
    KER_VER = (9, 1)
    KER_DEF = (3, 3)
    
    def __init__(self):
        super().__init__()
        stem = nn.Sequential(
            nn.Conv2d(self.IN_CON_CHANNELS, self.OUT_GEN_CONV, self.KER_DEF, padding='same'),
            nn.ReLU()
        )
        conv_general = nn.Sequential(
            ResBlock(self.OUT_GEN_CONV, self.OUT_GEN_CONV, self.KER_DEF),
            ResBlock(self.OUT_GEN_CONV, self.OUT_GEN_CONV, self.KER_DEF)
        )
        conv_striped = nn.Sequential(
            ResBlock(self.OUT_GEN_CONV, self.OUT_GEN_CONV, self.KER_LAT),
            ResBlock(self.OUT_GEN_CONV, self.OUT_GEN_CONV, self.KER_VER)
        )
        conv_summary = nn.Sequential(
            nn.Conv2d(self.OUT_GEN_CONV, self.OUT_SUM_CONV, self.KER_DEF, padding='same'),
            nn.ReLU(),
            nn.Flatten()
        )
        self.conv_net = nn.Sequential(
            stem,
            conv_general,
            conv_striped,
            conv_summary
        )

    def forward(self, x):
        return self.conv_net(x)

class BL_DQN_Action_Network_v1(nn.Module):
    STR_DENSE_DIM = 16
    OUT_EMBED_DIM = 16

    CONV_NET_OUT = BL_DQN_Conv_Network.OUT_SUM_CONV * bl.BOARD_FLAT
    IN_LIN_DIM = CONV_NET_OUT + (OUT_EMBED_DIM * bl.BLOCKS_TO_PICK) + STR_DENSE_DIM
    HID_LIN_DIM = 256

    def __init__(self):
        super().__init__()
        self.conv_net = BL_DQN_Conv_Network()
        self.embed_net = nn.Sequential(
            nn.Embedding(bl.BLOCKS_SIZE + 1, self.OUT_EMBED_DIM),
            nn.Flatten()
        )
        self.str_dense_net = nn.Sequential(
            nn.Linear(1, self.STR_DENSE_DIM),
            nn.ReLU()
        )

        self.afterwards_net = nn.Sequential(
            nn.Linear(self.IN_LIN_DIM, self.HID_LIN_DIM),
            nn.ReLU(),
            nn.Linear(self.HID_LIN_DIM, self.HID_LIN_DIM),
            nn.ReLU(),
            nn.Linear(self.HID_LIN_DIM, bl.NN_OUTPUT_FLAT),
        )

    def forward(self, board: torch.Tensor, blocks: torch.Tensor, streak: torch.Tensor):
        board_out = self.conv_net(board)
        blocks_out = self.embed_net(blocks)
        streak_out = self.str_dense_net(streak)

        final_input = torch.cat((board_out, blocks_out, streak_out), dim=1)
        return self.afterwards_net(final_input)

    def validate_model(self, num_games: int = 1):
        scores = []
        for _ in range(num_games):
            st_nw = bl.State()
            curr_mask = st_nw.calculate_mask()
            game_over = False
            score = 0
            while not game_over:
                board, blocks, streak = st_nw.in_board(), st_nw.in_blocks(), st_nw.in_streak()

                nn_board = torch.from_numpy(board).float().unsqueeze(0)
                nn_blocks = torch.from_numpy(blocks).long().unsqueeze(0)
                nn_streak = torch.from_numpy(streak).float().view(1, 1)

                with torch.no_grad():
                    y_hat_nn = self.forward(nn_board, nn_blocks, nn_streak).squeeze(0)
                    y_hat_nn[~curr_mask] = -1e9

                action = int(torch.argmax(y_hat_nn))
                piece, x, y = bl.State.decode_action(action)

                rwrd, is_terminal, new_mask = st_nw.transition(piece, x, y)
                game_over = is_terminal
                curr_mask = new_mask
                score += rwrd
            scores.append(score - bl.NEURAL_PENALTY)

        return f"Points in val games: {scores}, mean: {np.mean(scores):.2f}, std: {np.std(scores):.2f}"

class BL_DQN_State_Network(nn.Module):
    STR_DENSE_DIM = 16
    OUT_EMBED_DIM = 16

    CONV_NET_OUT = BL_DQN_Conv_Network.OUT_SUM_CONV * bl.BOARD_FLAT
    IN_LIN_DIM = CONV_NET_OUT + (OUT_EMBED_DIM * bl.BLOCKS_TO_PICK) + STR_DENSE_DIM
    HID_LIN_DIM = 256

    def __init__(self):
        super().__init__()
        self.conv_net = BL_DQN_Conv_Network()
        self.embed_net = nn.Sequential(
            nn.Embedding(bl.BLOCKS_SIZE + 1, self.OUT_EMBED_DIM),
            nn.Flatten()
        )
        self.str_dense_net = nn.Sequential(
            nn.Linear(1, self.STR_DENSE_DIM),
            nn.ReLU()
        )

        self.afterwards_net = nn.Sequential(
            nn.Linear(self.IN_LIN_DIM, self.HID_LIN_DIM),
            nn.ReLU(),
            nn.Linear(self.HID_LIN_DIM, self.HID_LIN_DIM),
            nn.ReLU(),
        )

    def forward(self, board: torch.Tensor, blocks: torch.Tensor, streak: torch.Tensor):
        board_out = self.conv_net(board)
        blocks_out = self.embed_net(blocks)
        streak_out = self.str_dense_net(streak)

        final_input = torch.cat((board_out, blocks_out, streak_out), dim=1)
        return self.afterwards_net(final_input)

class BL_DQN_State_Network_v1(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = BL_DQN_State_Network()
        self.final = nn.Linear(BL_DQN_State_Network.HID_LIN_DIM, 1)

    def forward(self, board: torch.Tensor, blocks: torch.Tensor, streak: torch.Tensor):
        x = self.network(board, blocks, streak)
        return self.final(x)

class BL_DQN_State_Network_v2(nn.Module):
    def __init__(self, num_aux_features: int = 0):
        super().__init__()
        self.network = BL_DQN_State_Network()
        self.value_head = nn.Linear(BL_DQN_State_Network.HID_LIN_DIM, 1)
        self.aux_head = nn.Linear(BL_DQN_State_Network.HID_LIN_DIM, num_aux_features)

    def forward(self, board: torch.Tensor, blocks: torch.Tensor, streak: torch.Tensor):
        x = self.network(board, blocks, streak)
        v_value = self.value_head(x)
        aux_features = self.aux_head(x)
        
        return v_value, aux_features

class ActionsStorage:
    TOTAL_SIZE = 1_000_000

    def __init__(self):
        self.boards_s = np.empty((self.TOTAL_SIZE, 1, bl.BOARD_LEN, bl.BOARD_LEN), dtype=bool)
        self.blocks_s = np.empty((self.TOTAL_SIZE, bl.BLOCKS_TO_PICK), dtype=np.int8)
        self.streaks_s = np.empty((self.TOTAL_SIZE, 1), dtype=bool)

        self.boards_f = np.empty((self.TOTAL_SIZE, 1, bl.BOARD_LEN, bl.BOARD_LEN), dtype=bool)
        self.blocks_f = np.empty((self.TOTAL_SIZE, bl.BLOCKS_TO_PICK), dtype=np.int8)
        self.streaks_f = np.empty((self.TOTAL_SIZE, 1), dtype=bool)

        self.actions = np.empty(self.TOTAL_SIZE, dtype=np.int16)
        self.rewards = np.empty(self.TOTAL_SIZE, dtype=np.int16)
        self.terminal = np.empty(self.TOTAL_SIZE, dtype=bool)
        self.masks_f = np.empty((self.TOTAL_SIZE, bl.NN_OUTPUT_FLAT), dtype=bool)

        self.offset = 0
        self.size = 0

    def _write_arrays(self, incoming_data: tuple, n: int):
        buffers = (
            self.boards_s, self.blocks_s, self.streaks_s,
            self.boards_f, self.blocks_f, self.streaks_f,
            self.actions, self.rewards, self.terminal, self.masks_f
        )

        if self.offset + n <= self.TOTAL_SIZE:
            for buf, data in zip(buffers, incoming_data):
                buf[self.offset : self.offset + n] = data
        else:
            space_left = self.TOTAL_SIZE - self.offset
            remainder = n - space_left

            for buf, data in zip(buffers, incoming_data):
                buf[self.offset : self.TOTAL_SIZE] = data[:space_left]
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
        self.offset = (self.offset + n) % self.TOTAL_SIZE
        self.size = min(self.size + n, self.TOTAL_SIZE)

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

class StateStorage:
    TOTAL_SIZE = 1_000_000
    def __init__(self):
        self.boards_s = np.empty((self.TOTAL_SIZE, 1, bl.BOARD_LEN, bl.BOARD_LEN), dtype=bool)
        self.blocks_s = np.empty((self.TOTAL_SIZE, bl.BLOCKS_TO_PICK), dtype=np.int8)
        self.streaks_s = np.empty((self.TOTAL_SIZE, 1), dtype=bool)

        self.boards_f = np.empty((self.TOTAL_SIZE, 1, bl.BOARD_LEN, bl.BOARD_LEN), dtype=bool)
        self.blocks_f = np.empty((self.TOTAL_SIZE, bl.BLOCKS_TO_PICK), dtype=np.int8)
        self.streaks_f = np.empty((self.TOTAL_SIZE, 1), dtype=bool)

        self.rewards = np.empty(self.TOTAL_SIZE, dtype=np.int16)
        self.terminal = np.empty(self.TOTAL_SIZE, dtype=bool)
        self.offset = 0
        self.size = 0

    def _write_arrays(self, incoming_data: tuple, n: int):
        buffers = (
            self.boards_s, self.blocks_s, self.streaks_s,
            self.boards_f, self.blocks_f, self.streaks_f,
            self.rewards, self.terminal
        )

        if self.offset + n <= self.TOTAL_SIZE:
            for buf, data in zip(buffers, incoming_data):
                buf[self.offset : self.offset + n] = data
        else:
            space_left = self.TOTAL_SIZE - self.offset
            remainder = n - space_left

            for buf, data in zip(buffers, incoming_data):
                buf[self.offset : self.TOTAL_SIZE] = data[:space_left]
                buf[0 : remainder] = data[space_left:]

    def add_records(self, boards_s: np.ndarray, blocks_s: np.ndarray, streaks_s: np.ndarray,
                          boards_f: np.ndarray, blocks_f: np.ndarray, streaks_f: np.ndarray,
                          rewards: np.ndarray, terminals: np.ndarray):
        n = len(boards_s) # Amount of records to add.
        incoming_data = (
            boards_s, blocks_s, streaks_s,
            boards_f, blocks_f, streaks_f,
            rewards, terminals
        )

        self._write_arrays(incoming_data, n)
        self.offset = (self.offset + n) % self.TOTAL_SIZE
        self.size = min(self.size + n, self.TOTAL_SIZE)

    def sample(self, num_samples: int = 1):
        idx = bl.rng_fact_instance.sample(range(self.size), num_samples)
        return (
            self.boards_s[idx],
            self.blocks_s[idx],
            self.streaks_s[idx],
            self.boards_f[idx],
            self.blocks_f[idx],
            self.streaks_f[idx],
            self.rewards[idx],
            self.terminal[idx],
        )

class SmartStateStorage:
    EMPTY_BOARDS_MAX = 16
    CASUAL_BOARDS_MAX = 32
    CROWDED_BOARDS_MAX = 48

    def __init__(self):
        self.empty_boards = StateStorage()
        self.casual_boards = StateStorage()
        self.crowded_boards = StateStorage()
        self.ultra_crowded_boards = StateStorage()

    @property
    def size(self):
        return min(self.empty_boards.size, 
                   self.casual_boards.size, 
                   self.crowded_boards.size, 
                   self.ultra_crowded_boards.size)

    def add_records(self, boards_s: np.ndarray, blocks_s: np.ndarray, streaks_s: np.ndarray,
                          boards_f: np.ndarray, blocks_f: np.ndarray, streaks_f: np.ndarray,
                          rewards: np.ndarray, terminals: np.ndarray):
        board_counts = np.sum(boards_s.reshape(len(boards_s), -1), axis=1)
        empty_mask = board_counts <= self.EMPTY_BOARDS_MAX
        casual_mask = (board_counts > self.EMPTY_BOARDS_MAX) & (board_counts <= self.CASUAL_BOARDS_MAX)
        crowded_mask = (board_counts > self.CASUAL_BOARDS_MAX) & (board_counts <= self.CROWDED_BOARDS_MAX) 
        ultra_crowded_mask = board_counts > self.CROWDED_BOARDS_MAX

        def push(storage_obj, mask):
            if np.any(mask):
                storage_obj.add_records(
                    boards_s[mask], blocks_s[mask], streaks_s[mask],
                    boards_f[mask], blocks_f[mask], streaks_f[mask],
                    rewards[mask], terminals[mask]
                )
        push(self.empty_boards, empty_mask)
        push(self.casual_boards, casual_mask)
        push(self.crowded_boards, crowded_mask)
        push(self.ultra_crowded_boards, ultra_crowded_mask)

    def sample(self, num_samples: int = 1):
        total_parts = 8
        samples = [i * num_samples // total_parts for i in [1, 3, 3]]

        s1 = self.empty_boards.sample(samples[0])
        s2 = self.casual_boards.sample(samples[1])
        s3 = self.crowded_boards.sample(samples[2])
        s4 = self.ultra_crowded_boards.sample(num_samples - sum(samples))
        combined_data = tuple (
            np.concatenate(arrays, axis=0) for arrays in zip(s1, s2, s3, s4)
        )
        return combined_data