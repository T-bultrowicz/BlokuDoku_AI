import blokudoku_lib as bl

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """Build single residual convolutional network block.

    Args:
        channels: nn.conv2d in_channels parameter.
        ker: nn.conv2d kernel_size parameter.
    """
    def __init__(self, channels: int, ker: tuple):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, 
                               kernel_size=ker, padding='same')
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, 
                               kernel_size=ker, padding='same')

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(out + residual)

class BL_DQN_Conv_Network(nn.Module):
    """Build conv_network responsible for analyzing board state.

    Paramter count: 314 528. Activation: ReLU
    Args:
        in_chan: Number of input conv2d channels.
        out_gen: Number of conv2d channels in hidden layers.
        out_sum: Number of conv2d channels in output layer.
    """
    IN_CON_CHANNELS = 1
    OUT_GEN_CONV = 64
    OUT_SUM_CONV = 32
    
    KER_LAT = (1, 9)
    KER_VER = (9, 1)
    KER_DEF = (3, 3)
    
    def __init__(self, in_chan: int = IN_CON_CHANNELS, 
                        out_gen: int = OUT_GEN_CONV, 
                        out_sum: int = OUT_SUM_CONV):
        super().__init__()
        stem = nn.Sequential(
            nn.Conv2d(in_chan, out_gen, self.KER_DEF, padding='same'),
            nn.ReLU()
        )
        conv_general = nn.Sequential(
            ResBlock(out_gen, self.KER_DEF),
            ResBlock(out_gen, self.KER_DEF)
        )
        conv_striped = nn.Sequential(
            ResBlock(out_gen, self.KER_LAT),
            ResBlock(out_gen, self.KER_VER)
        )
        conv_summary = nn.Sequential(
            nn.Conv2d(out_gen, out_sum, self.KER_DEF, padding='same'),
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

class BL_DQN_Conv_Network_Light(BL_DQN_Conv_Network):
    """ Build lighter version of BL_DQN_CONV_Network.

    Parameter count: 177 144. Activation: ReLU
    """
    IN_CON_CHANNELS = 1
    OUT_GEN_CONV = 48
    OUT_SUM_CONV = 24
    
    def __init__(self):
        super().__init__(self.IN_CON_CHANNELS, self.OUT_GEN_CONV, self.OUT_SUM_CONV)

class BL_DQN_Conv_Network_Heavy(nn.Module):
    """ Build heavier version of BL_DQN_CONV_Network.

    This network has high possibility of overfitting.
    Parameter count: 1 846 080 Activation: ReLU
    """
    IN_CON_CHANNELS = 1
    OUT_GEN_CONV = 128
    OUT_SUM_CONV = 64

    KER_LAT = (1, 9)
    KER_VER = (9, 1)
    KER_DEF = (3, 3)

    def __init__(self):
        super().__init__()

        in_chan = self.IN_CON_CHANNELS
        out_gen = self.OUT_GEN_CONV
        out_sum = self.OUT_SUM_CONV
        stem = nn.Sequential(
            nn.Conv2d(in_chan, out_gen, self.KER_DEF, padding='same'),
            nn.ReLU()
        )
        conv_general = nn.Sequential(
            ResBlock(out_gen, self.KER_DEF),
            ResBlock(out_gen, self.KER_DEF)
        )
        conv_striped = nn.Sequential(
            ResBlock(out_gen, self.KER_LAT),
            ResBlock(out_gen, self.KER_VER),
            ResBlock(out_gen, self.KER_LAT),
            ResBlock(out_gen, self.KER_VER)
        )
        conv_summary = nn.Sequential(
            nn.Conv2d(out_gen, out_sum, self.KER_DEF, padding='same'),
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
    """ Build NN working as regressor for Belmann Equations of Bloku Doku game.

    This NN achieves very poor results.
    """
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
        """ Validate perfomance of NN.

        Args:
            num_games: Number of games to validate on.
        """
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

class BL_DQN_State_Network_Base(nn.Module):
    """ Build base for NN models designed to be predictors for values of states.

    Args:
        conv_class: Convolutional network for board state analysis.
        conv_out_channels: conv_out parameter of passed in conv_class.
        hid_dim_1: Neuron count in first hidden later.
        hid_dim_2: Neuron count in second hidden later.
    """
    STR_DENSE_DIM = 16
    OUT_EMBED_DIM = 16
    HID_LIN_DIM_1 = 256
    HID_LIN_DIM_2 = 256

    def __init__(self, conv_class, conv_out_channels, 
                hid_dim_1: int = HID_LIN_DIM_1, hid_dim_2: int = HID_LIN_DIM_2):
        super().__init__()
        self.conv_net = conv_class()
        
        self.embed_net = nn.Sequential(
            nn.Embedding(bl.BLOCKS_SIZE + 1, self.OUT_EMBED_DIM),
            nn.Flatten()
        )
        self.str_dense_net = nn.Sequential(
            nn.Linear(1, self.STR_DENSE_DIM),
            nn.ReLU()
        )

        in_lin_dim = (conv_out_channels * bl.BOARD_FLAT) + \
                     (self.OUT_EMBED_DIM * bl.BLOCKS_TO_PICK) + \
                     self.STR_DENSE_DIM

        self.afterwards_net = nn.Sequential(
            nn.Linear(in_lin_dim, hid_dim_1),
            nn.ReLU(),
            nn.Linear(hid_dim_1, hid_dim_2),
            nn.ReLU(),
        )

    def forward(self, board: torch.Tensor, blocks: torch.Tensor, streak: torch.Tensor):
        board_out = self.conv_net(board)
        blocks_out = self.embed_net(blocks)
        streak_out = self.str_dense_net(streak)

        final_input = torch.cat((board_out, blocks_out, streak_out), dim=1)
        return self.afterwards_net(final_input)

class BL_DQN_State_Network(BL_DQN_State_Network_Base):
    """ Build basic NN predictor for state values. Achieves the best results.

    Parameter count: 314 528 (CNN) + 746 784 (Dense) = 1 061 312
    Activation: ReLU
    """
    def __init__(self):
        super().__init__(BL_DQN_Conv_Network, BL_DQN_Conv_Network.OUT_SUM_CONV)

class BL_DQN_State_Network_Light(BL_DQN_State_Network_Base):
    """ Build light version NN predictor for state values.

    Parameter count: 177 144 (CNN) + 521 264 (Dense) = 698 408
    Activation: ReLU
    """
    STR_DENSE_DIM = 8
    HID_LIN_DIM_1 = 256
    HID_LIN_DIM_2 = 32

    def __init__(self):
        super().__init__(BL_DQN_Conv_Network_Light, BL_DQN_Conv_Network_Light.OUT_SUM_CONV, 
                    self.HID_LIN_DIM_1, self.HID_LIN_DIM_2)

class BL_DQN_State_Network_Heavy(BL_DQN_State_Network_Base):
    """ Build heavy version NN predictor for state values.

    Paramter count: 1 846 080 (CNN) + 1 410 336 (Dense) = 3 256 416
    Activation: ReLU
    Currently overfits! Forget or try to adjust learning hyperparameters.
    """
    HID_LIN_DIM_1 = 256
    HID_LIN_DIM_2 = 256

    def __init__(self):
        super().__init__(BL_DQN_Conv_Network_Heavy, BL_DQN_Conv_Network_Heavy.OUT_SUM_CONV,
                    self.HID_LIN_DIM_1, self.HID_LIN_DIM_2)

class BL_DQN_State_Network_v1(nn.Module):
    """ Constructor for different NNs with no auxiliary features prediction.
    """
    def __init__(self, 
    base_network: BL_DQN_State_Network | BL_DQN_State_Network_Light | BL_DQN_State_Network_Heavy):
        super().__init__()
        self.network = base_network
        self.final = nn.Linear(base_network.HID_LIN_DIM_2, 1)

    def forward(self, board: torch.Tensor, blocks: torch.Tensor, streak: torch.Tensor):
        x = self.network(board, blocks, streak)
        return self.final(x)

class BL_DQN_State_Network_v2(nn.Module): 
    """ Constructor for different NNs with additional auxiliary features prediction.
    """
    def __init__(self, 
                base_network: BL_DQN_State_Network | BL_DQN_State_Network_Light | BL_DQN_State_Network_Heavy,
                num_aux_features: int = 0):
        super().__init__()
        self.network = base_network
        self.value_head = nn.Linear(base_network.HID_LIN_DIM_2, 1)
        self.aux_head = nn.Linear(base_network.HID_LIN_DIM_2, num_aux_features)

    """ Forward pass of neural network.

    Returns:
        Vs: Predicted values of passed in states.
        Auxs: Predicted valeus of auxilliary features, that passed in states have.
    """
    def forward(self, board: torch.Tensor, blocks: torch.Tensor, streak: torch.Tensor):
        x = self.network(board, blocks, streak)
        v_value = self.value_head(x)
        aux_features = self.aux_head(x)
        
        return v_value, aux_features

class ActionsStorage:
    """ Class for optimal, low memory storage of moves history.

    Works as circular buffer with memory off up to self.TOTAL_SIZE (1e6) games.
    Automatically forgets the oldest moves when it gets filled up.
    """
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
        """ Add detailed records of singular moves to storage.

        Args:
            board_s, blocks_s, streaks_s: Pre-action state.
            boards_f, blocks_f, streaks_f: Post-action state.
            actions: Actions encoded as integers.
            rewards: Rewards from taking actions.
            terminals, masks_f: Whether post-action state is terminal and possible moves from it.
        """
        
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
        """ Randomly sample games from storage.

        Args:
            num_samples: Size of the sample.

        Returns:
            np.arrays representing informations of singular moves, in a same
            order and functionality as described for add_records() method.
        """
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
    """ Class for optimal, low memory storage of moves history.

    Works as circular buffer with memory off up to self.TOTAL_SIZE (1e6) games.
    Automatically forgets the oldest moves when it gets filled up.
    """
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
        """ Add detailed records of singular moves to storage.

        Args:
            board_s, blocks_s, streaks_s: Pre-action state.
            boards_f, blocks_f, streaks_f: Post-action state.
            actions: Actions encoded as integers.
            rewards: Rewards from taking actions.
            terminals: Whether post-action state is terminal.
        """
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
        """ Randomly sample games from storage.

        Args:
            num_samples: Size of the sample.

        Returns:
            np.arrays representing informations of singular moves, in a same
            order and functionality as described for add_records() method.
        """
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
    """ Class for optimal, low memory storage of moves history, that allows
    sampling with respect to stratified distribution based on boards crowdness.

    Works as circular buffer with memory off up to self.TOTAL_SIZE (1e6) games.
    Automatically forgets the oldest moves when it gets filled up.
    EXPERIMENTAL, DID NOT ACHIEVE ANY PROFOUND RESULTS.
    """
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
        """ Add detailed records of singular moves to storage.

        Args:
            board_s, blocks_s, streaks_s: Pre-action state.
            boards_f, blocks_f, streaks_f: Post-action state.
            actions: Actions encoded as integers.
            rewards: Rewards from taking actions.
            terminals: Whether post-action state is terminal.
        """
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
        """ Randomly sample games from storage, following stratified distribution
        based around crowdness of the board.

        Args:
            num_samples: Size of the sample.

        Returns:
            np.arrays representing informations of singular moves, in a same
            order and functionality as described for add_records() method.
        """
        part = 5 * num_samples // 16

        s1 = self.empty_boards.sample(part)
        s2 = self.casual_boards.sample(part)
        s3 = self.crowded_boards.sample(part)
        s4 = self.ultra_crowded_boards.sample(num_samples - 3 * part)
        combined_data = tuple (
            np.concatenate(arrays, axis=0) for arrays in zip(s1, s2, s3, s4)
        )
        return combined_data