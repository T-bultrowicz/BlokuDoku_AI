import blokudoku_lib as bl
import network_lib as net
import user_play as user_play

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import profiling
from contextlib import contextmanager
from pathlib import Path

# Weights and best scores live here, no matter the current working directory.
NETWORKS_DIR = Path(__file__).resolve().parents[1] / "networks"
NETWORKS_DIR.mkdir(exist_ok=True)

# DEVICE DEPENDENT PARAMETRES
(GAMES_IN_PARALLEL, EXCHANGE_NETS_FREQ, BATCH_SIZE) = (
    (64, 60, 512) if bl.DEVICE.type == "cuda" else (16, 250, 128)
)

# LONGEVITY OF TRAINING AND VALIDATION INTERVAL PARAMETERS
GAMES_TO_PLAY_TRAIN = 60_000
GAMES_TO_PLAY_TEST = 500
GAMES_TO_PLAY_PROFILE = 3_000
DISPLAY_ERR_FREQ = 1500
GAMES_TO_DISPLAY_INFO = 250
VAL_GAMES_CHUNK = 256  # Chunk size for val_net.
VALIDATION_GAMES_PER_BATCH = 5
VALIDATION_GAMES_PER_FIGHT = 25

# BASIC LEARING HYPERPARAMETERS
EPS_MAX = 1.0
EPS_MIN = 0.005
DECAY_RATE = 0.9999975
DISCOUNT = 0.995
LEARING_OPTIM_RATE = 1e-4

# ADVANCED OPTIONS
MARGIN = 50.0
SIDE_TASK_WEIGHT = 7.0
EPS_THRESHOLD = 0.15
STEP_TO_MODIFY = 15000
GAMMA = 0.5


def try_write_best_net(
    netw: nn.Module,
    filepath_net: str | Path,
    filepath_score: str | Path,
    val_games_num: int,
    netw_res: float = 0.0,
):
    """Save network and results it achieved.

    Used as validation tool in model selection. Be cautious to invoke it
    only when same type of network is scored in file filepath_net, otherwise
    may result in fatal error. Follows algorithm below:

    1. If there is no proper save of network under filepath_net,
    save netw and netw_res (return).

    2. Otherwise, compare netw_res with score on filepath_score, if netw_res is
    worse more than MARGIN points, forfeit saving to not overwrite better model.

    3. If scores are similar, perform additional validation on val_games_num
    games to save better performing net.

    P.S. If filepath_score is not present, validation of both network still occurs.

    Args:
        netw: network model to save
        filepath_net: filepath to save netw
        filepath_score: filepath to save netw result
        val_games_num: number of validation games to perform, if needed.
        netw_res: result network achieved when validated first.
    """
    if not os.path.exists(filepath_net):
        torch.save(netw.state_dict(), filepath_net)
        current_score, _ = val_net(netw, val_games_num)  # type: ignore

        with open(filepath_score, "w") as f:
            f.write(str(current_score))
        print(f"Base score set: {current_score:.2f}")
        return

    with open(filepath_score, "r") as f:
        try:
            saved_score = float(f.read())
        except ValueError:
            saved_score = netw_res
    if netw_res + MARGIN < saved_score:
        return

    saved_net = copy.deepcopy(netw)
    load_success = False
    try:
        net.load_weights(saved_net, filepath_net)
        load_success = True
    except Exception as e:
        load_success = False

    if not load_success:
        torch.save(netw.state_dict(), filepath_net)
        current_score, _ = val_net(netw, val_games_num)  # type: ignore

        with open(filepath_score, "w") as f:
            f.write(str(current_score))
        print(f"New best score: {current_score:.2f}")
        return

    print(f"\nStarting game validation - two agent fight ...")
    current_score, _ = val_net(netw, val_games_num)  # type: ignore
    saved_score, _ = val_net(saved_net, val_games_num)  # type: ignore

    # Kto wygrał?
    if current_score > saved_score:
        print(
            f"We got new champion! - result of: {current_score:.2f} vs {saved_score:.2f}"
        )
        torch.save(netw.state_dict(), filepath_net)
        with open(filepath_score, "w") as f:
            f.write(str(current_score))
    else:
        print(f"Saved net held - result of: {current_score:.2f} vs {saved_score:.2f}")
        with open(filepath_score, "w") as f:
            f.write(str(saved_score))

def _segment_argmax(vals: torch.Tensor, gid: torch.Tensor, n: int):
    """Calculate argmax of vals separately inside every game.

    Args:
        vals: (M,) values of stacked moves.
        gid: (M,) long, game index of every move.
        n: number of games, each must own at least one move.

    Returns:
        best: (n,) long, row of the first maximal value of every game.
    """
    # Get best values.
    M = len(vals)
    best_vals = torch.full((n,), -torch.inf, dtype=vals.dtype, device=vals.device)
    best_vals.scatter_reduce_(0, gid, vals, reduce="amax")
    # Revocer indices of best values.
    rows = torch.arange(M, device=vals.device)
    candidates = torch.where(vals == best_vals[gid], rows, M)
    best = torch.full((n,), M, dtype=torch.long, device=vals.device)
    return best.scatter_reduce(0, gid, candidates, reduce="amin")


def _eps_greedy_rows(qs: torch.Tensor, lengths: torch.Tensor, eps: float):
    """Choose one move per game: the best one, or with probability eps a random legal one.

    Args:
        qs: (M,) values of stacked legal moves, game-major.
        lengths: (N,) long, legal moves count per game, each > 0.

    Returns:
        rows: (N,) long, chosen row for every game.
    """
    N, M = len(lengths), len(qs)
    pos = torch.arange(N, device=bl.DEVICE)
    best = _segment_argmax(qs, torch.repeat_interleave(pos, lengths, output_size=M), N)
    if eps == 0:
        return best
    random = lengths.cumsum(0) - lengths + (torch.rand(N, device=bl.DEVICE) * lengths).long()
    return torch.where(torch.rand(N, device=bl.DEVICE) < eps, random, best)


def _move_values(netw: nn.Module, vs: bl.VectorizedStates, A, B, L, S, MC):
    """Values netw assigns to stacked legal moves of vs: V(afterstate) for state
    networks, Q(state, action) for action networks.

    Args:
        A, B, L, S, MC: as returned by vs.states_prim().

    Returns:
        (M,) floats, one per row of states_prim().
    """
    with torch.no_grad():
        if isinstance(netw, net.BL_DQN_Action_Network_v1):
            boards = vs.boards.view(-1, 1, bl.BOARD_LEN, bl.BOARD_LEN).float()
            qs = netw(boards, vs.available.long(), vs.streaks.float())  # (N, 243)
            pos = torch.arange(vs.N, device=bl.DEVICE)
            return qs[torch.repeat_interleave(pos, MC, output_size=len(A)), A]
        out = netw(B.float(), L.long(), S.float())
    return (out[0] if isinstance(out, tuple) else out).view(-1)


def _play_greedy_batch(netw: nn.Module, games: int):
    """Play `games` games in parallel, greedily, until every one of them ends.

    Returns:
        (games,) int32 tensor of summed rewards, NEURAL_PENALTY included.
    """
    scores = torch.zeros(games, dtype=torch.int32, device=bl.DEVICE)
    game_ids = torch.arange(games, device=bl.DEVICE)  # original index of live games
    vs = bl.VectorizedStates(games)
    while vs.N > 0:
        A, R, T, B, L, S, MC = vs.states_prim()
        best = _eps_greedy_rows(_move_values(netw, vs, A, B, L, S, MC), MC, 0.0)
        scores[game_ids] += R[best]
        vs.mass_transition(B[best], L[best], S[best])
        vs.drop(T[best])
        game_ids = game_ids[~T[best]]
    return scores


def val_net(
    netw: nn.Module,
    games_count: int = 100,
    more_info: bool = False,
    games_chunk: int = VAL_GAMES_CHUNK,
):
    """Validate netw performance on games_count games, played greedily in batches.

    Args:
        games_count: games to play
        more_info: Whether to return max and min of achieved results.
        games_chunk: Games played in parallel, bounds memory for big games_count.

    Returns:
        mean: mean of achieved results
        std: std of achieved results
        max (only if more_info=True): max of achieved results
        min (only if more_info=True): min of achieved results
    """
    was_training = netw.training
    netw.eval()
    scores = torch.cat(
        [
            _play_greedy_batch(netw, min(games_chunk, games_count - s))
            for s in range(0, games_count, games_chunk)
        ]
    )
    netw.train(was_training)

    scores = (scores - bl.NEURAL_PENALTY).float()
    mean, std = scores.mean().item(), scores.std(correction=0).item()
    if more_info:
        return mean, std, scores.max().item(), scores.min().item()
    return mean, std


def calculate_aux_targets(boards_tensor: torch.Tensor):
    """Calculate auxiliray targets for a boards_tensor,
    representing look of boards.

    Returns:
        torch.Tensor of shape (Batch_num, Aux_feat_num), where aux features include:
        1. Sum of taken fileds.
        2. Non empty columns counter.
        3. Non emprt rows counter.
        4. Non empty 3x3 blocks counter.
        5. Number of holes inside a board (empty places fully surrounded by blocks).
    """
    b = boards_tensor.squeeze(1).bool()

    # 1. Taken fields
    f1 = b.sum(dim=(1, 2))
    # 2. Non empty columns (normalized 0-1)
    f2 = (b.sum(dim=1) > 0).float().sum(dim=1)
    # 3. Non empty rows (normalized 0-1)
    f3 = (b.sum(dim=2) > 0).float().sum(dim=1)
    # 4. Non empty 3x3 blocks (normalized 0-1)
    f4 = torch.stack(
        [
            b[:, i : i + 3, j : j + 3].sum(dim=(1, 2)) > 0
            for i in range(0, 9, 3)
            for j in range(0, 9, 3)
        ],
        dim=1,
    ).sum(dim=1)
    # 5. Holes in the board (normalized 0 - 1)
    tmp = torch.ones_like(b)
    # So tmp contains only fields surrounded by True fields.
    tmp[:, :, 1:] &= b[:, :, :-1]
    tmp[:, 1:, :] &= b[:, :-1, :]
    tmp[:, :, :-1] &= b[:, :, 1:]
    tmp[:, :-1, :] &= b[:, 1:, :]
    # Now every empty field, but surrounded, is a hole.
    holes = (~b) & tmp
    f5 = holes.sum(dim=(1, 2)).float()

    return torch.stack([f1, f2, f3, f4, f5], dim=1)


class ActionNetworkTrainer:
    """Class capable of triaing and testing BL_DQN_Action_Network_v1().

    Defaultly saves network under path ActionNetworkTrainer.SAVE_PATH,
    feel free to overwrite if different path is desired.
    """

    SAVE_PATH = NETWORKS_DIR / "action_network_v1.pt"

    def showcase_network(self):
        raise RuntimeError(
            "Showcase network is not yet implemented for action network! \
(Sorry for the inconvenience)"
        )

    def test_network(self, games_to_play: int = GAMES_TO_PLAY_TEST):
        """Tests netowork on games_to_play games."""
        main_net = net.BL_DQN_Action_Network_v1().to(bl.DEVICE)
        try:
            net.load_weights(main_net, self.SAVE_PATH)
        except Exception as e:
            print(
                f"Error - didn't managed to find network params in filepath{self.SAVE_PATH}\n"
            )
            return

        score, score_std = val_net(main_net, games_to_play)  # type: ignore
        print(f"Tested model performance - mean score: {score:.2f}, std: {score_std:.2f}")

    def _play_parallel_games(
        self,
        states: bl.VectorizedStates,
        main_net: net.BL_DQN_Action_Network_v1,
        storage: net.ActionsStorage,
        eps: float,
    ):
        """Make one move in every game, eps-greedily according to main_net.
        Moves are saved into storage, finished games start anew.

        Returns:
            terminals: (N,) bools, whether game ended with its move.
        """
        A, R, T, B, L, S, MC = states.states_prim()
        boards_s, blocks_s, streaks_s = states.boards, states.available, states.streaks
        rows = _eps_greedy_rows(_move_values(main_net, states, A, B, L, S, MC), MC, eps)

        states.mass_transition(B[rows], L[rows], S[rows])
        # fmt: off
        storage.add_records(boards_s, blocks_s, streaks_s, B[rows], L[rows], S[rows],
                            A[rows], R[rows], T[rows], states.legal_moves())
        # fmt: on
        states.reset(T[rows])
        return T[rows]

    def _perform_gradient_descent(
        self,
        main_net: net.BL_DQN_Action_Network_v1,
        target_net: net.BL_DQN_Action_Network_v1,
        storage: net.ActionsStorage,
        optimizer: torch.optim.Adam,
    ):
        if storage.size < BATCH_SIZE:
            return

        # fmt: off
        (in_brd_st1, in_blk_st1, in_str_st1, 
         in_brd_st2, in_blk_st2, in_str_st2,
         b_actions, b_rewards, b_terminals, b_masks_f) = storage.sample(BATCH_SIZE)
        # fmt: on
        b_terminals = b_terminals.float()
        b_rewards = b_rewards.float()

        y_hat_nn_main = main_net(
            in_brd_st1.float(), in_blk_st1.long(), in_str_st1.float()
        )
        q_pred_fn = y_hat_nn_main[torch.arange(BATCH_SIZE), b_actions.long()]

        with torch.no_grad():
            y_hat_nn_target = target_net(
                in_brd_st2.float(), in_blk_st2.long(), in_str_st2.float()
            )
            y_hat_nn_target[~b_masks_f] = -1e9
            target_moves = torch.max(y_hat_nn_target, dim=1).values

        target_fn = b_rewards + (1 - b_terminals) * DISCOUNT * target_moves
        losses = F.mse_loss(q_pred_fn, target_fn)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    def _display_info(
        self,
        games_played: int,
        games_played_mod: int,
        batch_start_time: float,
        eps: float,
        main_net: net.BL_DQN_Action_Network_v1,
    ):
        if games_played_mod < GAMES_TO_DISPLAY_INFO:
            return games_played_mod, batch_start_time

        elapsed = time.perf_counter() - batch_start_time
        games_per_second = GAMES_TO_DISPLAY_INFO / elapsed if elapsed else float("inf")
        print(
            f"Games played: {games_played}, Games per second: "
            + f"{games_per_second:.2f}, batch time: {elapsed:.2f}s, eps: {eps:.4f}"
        )
        print(f"Validating model performance...")
        score, score_std = val_net(main_net, VALIDATION_GAMES_PER_BATCH)  # type: ignore
        print(f"Points in val games - mean: {score:.2f}, std: {score_std:.2f}\n")

        return games_played_mod % GAMES_TO_DISPLAY_INFO, time.perf_counter()

    def train_action_network(
        self, games_to_play: int = GAMES_TO_PLAY_TRAIN, games_count: int = GAMES_IN_PARALLEL
    ):
        """Trains network for games_to_play games, games_count of them being played in parallel.

        IMPORTANT: Changing games_count also requires to change global variable
        BATCH_SIZE to preserve same ratio of forward_moves / learning_moves ratio.
        Games_count equal 16 was tested to be optimal for speed on my device.

        """
        main_net = net.BL_DQN_Action_Network_v1().to(bl.DEVICE)
        target_net = net.BL_DQN_Action_Network_v1().to(bl.DEVICE)
        storage = net.ActionsStorage()

        optimizer = torch.optim.Adam(main_net.parameters(), lr=LEARING_OPTIM_RATE)
        batch_start_time = time.perf_counter()
        eps = EPS_MAX

        states = bl.VectorizedStates(games_count)
        games_played = 0
        games_played_mod = 0
        loops = 0
        while games_played < games_to_play:
            terminals = self._play_parallel_games(states, main_net, storage, eps)
            self._perform_gradient_descent(main_net, target_net, storage, optimizer)

            loops += 1
            eps = max(EPS_MIN, eps * (DECAY_RATE**games_count))
            term_sum = int(terminals.sum())
            games_played += term_sum
            games_played_mod += term_sum

            # EXCHANGE NETWORKS
            if loops % EXCHANGE_NETS_FREQ == 0:
                target_net.load_state_dict(main_net.state_dict())

            games_played_mod, batch_start_time = self._display_info(
                games_played, games_played_mod, batch_start_time, eps, main_net
            )

        torch.save(main_net.state_dict(), self.SAVE_PATH)


class StateNetworkTrainer:
    """Class for working with default BL_DQN_State_Network_v1() NN model."""

    SAVE_PATH_NET = NETWORKS_DIR / "state_network_v1.pt"
    SAVE_PATH_SCORE = NETWORKS_DIR / "best_score.txt"
    SHOWCASE_DELAY = 2  # Seconds between moves displayed by showcase_network.

    def showcase_network(self):
        """Interactive display in terminal (stdout), of how network saved under
        SAVE_PATH_NET plays the game.
        """
        main_net, _, _ = self._get_components()
        try:
            net.load_weights(main_net, self.SAVE_PATH_NET)
        except Exception as e:
            print(
                f"Error - didn't managed to find network params in filepath{self.SAVE_PATH_NET}\n"
            )
            return

        main_net.eval()
        score = 0
        game_over = False
        state = bl.State()
        while not game_over:
            user_play.output_to_player(state, score, game_over)
            _, R, T, B, L, S, MC = self._gather_data(state._vs)

            q_per_move = self._forward_data(R, T, B, L, S, main_net)
            row = self._agent_decide(q_per_move, MC, 0.0)
            state._vs.mass_transition(B[row], L[row], S[row])
            score += int(R[row])
            game_over = bool(T[row])
            time.sleep(self.SHOWCASE_DELAY)

        user_play.output_to_player(state, score - bl.NEURAL_PENALTY, game_over)

    def test_network(self, games_to_play: int = GAMES_TO_PLAY_TEST):
        """Validate network on the set of games_to_play games. Print result to stdout."""
        main_net, _, _ = self._get_components()
        try:
            net.load_weights(main_net, self.SAVE_PATH_NET)
        except Exception as e:
            print(
                f"Error - didn't managed to find network params in filepath{self.SAVE_PATH_NET}\n"
            )
            return

        score, score_std, max, min = val_net(main_net, games_to_play, True)  # type: ignore
        print(
            f"Tested model performance - mean score: {score:.2f}, std: {score_std:.2f}, \
max:{max:.4f}, min:{min:.4f}"
        )

    def _get_components(self):
        return (
            net.BL_DQN_State_Network_v1(net.BL_DQN_State_Network()).to(bl.DEVICE),
            net.BL_DQN_State_Network_v1(net.BL_DQN_State_Network()).to(bl.DEVICE),
            net.StateStorage(),
        )

    def _get_optimizer(self, net: nn.Module, lr: float, weight_decay: float = 1e-4):
        return torch.optim.Adam(net.parameters(), lr=lr)

    def _gather_data(self, states: bl.VectorizedStates):
        """Calculate all legal moves of every game and their afterstates.

        Returns:
            Tuple (A, R, T, B, L, S, MC), as described in VectorizedStates.states_prim().
        """
        return states.states_prim()

    def _forward_data(
        self,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        boards: torch.Tensor,
        blocks: torch.Tensor,
        streaks: torch.Tensor,
        nn_net: nn.Module,
    ):
        """Calculate Q values of moves: reward + discounted value nn_net predicts
        for their afterstates (terminal afterstates are worth 0).

        Returns:
            q_values: (M,) floats, one per passed afterstate.
        """
        with torch.no_grad():
            out = nn_net(boards.float(), blocks.long(), streaks.float())
            v_values = out[0].squeeze(1) if isinstance(out, tuple) else out.squeeze(1)

        q_values = rewards + ~terminals * DISCOUNT * v_values
        return q_values

    def _agent_decide(self, qs: torch.Tensor, lengths: torch.Tensor, eps: float):
        """Choose move in every game, best one or with probability eps random legal one.

        Returns:
            rows: (N,) long, row of chosen move for every game.
        """
        return _eps_greedy_rows(qs, lengths, eps)

    def _env_update(
        self,
        states: bl.VectorizedStates,
        storage,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        boards: torch.Tensor,
        blocks: torch.Tensor,
        streaks: torch.Tensor,
    ):
        """Save chosen moves into storage and move games to their afterstates.

        Finished games are started anew. Every argument is one per game.

        Returns:
            term_sum: number of games that ended.
        """
        # fmt: off
        storage.add_records(states.boards, states.available, states.streaks,
                            boards, blocks, streaks, rewards, terminals)
        # fmt: on
        states.mass_transition(boards, blocks, streaks)
        states.reset(terminals)
        return int(terminals.sum())

    def _perform_gradient_descent(
        self,
        main_net: nn.Module,
        target_net: nn.Module,
        storage,
        optimizer: torch.optim.Adam,
        display_err: bool = False,
    ):
        if storage.size < BATCH_SIZE:
            return

        # fmt: off
        (in_brd_st1, in_blk_st1, in_str_st1,
         in_brd_st2, in_blk_st2, in_str_st2,
         b_rewards, b_terminals) = storage.sample(BATCH_SIZE)
        # fmt: on
        b_terminals = b_terminals.float()
        b_rewards = b_rewards.float()

        y_hat_nn_main = main_net(
            in_brd_st1.float(), in_blk_st1.long(), in_str_st1.float()
        ).squeeze(1)
        with torch.no_grad():
            y_hat_nn_target = target_net(
                in_brd_st2.float(), in_blk_st2.long(), in_str_st2.float()
            ).squeeze(1)

        target_fn = b_rewards + (1 - b_terminals) * DISCOUNT * y_hat_nn_target
        losses = F.mse_loss(y_hat_nn_main, target_fn)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    def _display_info(
        self, games: int, games_mod: int, start: float, eps: float, main_net: nn.Module
    ):
        if games_mod < GAMES_TO_DISPLAY_INFO or games == 0:
            return games_mod, start

        elapsed = time.perf_counter() - start
        games_per_second = GAMES_TO_DISPLAY_INFO / elapsed if elapsed else float("inf")
        print(
            f"Games played: {games}, Games per second: "
            + f"{games_per_second:.2f}, batch time: {elapsed:.2f}s, eps: {eps:.4f}"
        )
        print(f"Validating model performance...")

        score, score_std = val_net(main_net, VALIDATION_GAMES_PER_BATCH)  # type: ignore
        print(f"Points in val games - mean: {score:.2f}, std: {score_std:.2f}")
        print()

        try_write_best_net(
            main_net,
            self.SAVE_PATH_NET,
            self.SAVE_PATH_SCORE,
            VALIDATION_GAMES_PER_FIGHT,
            score,
        )
        return games_mod % GAMES_TO_DISPLAY_INFO, time.perf_counter()

    def train_state_network(
        self,
        games_to_play: int = GAMES_TO_PLAY_TRAIN,
        games_count: int = GAMES_IN_PARALLEL,
    ):
        """Trains network for games_to_play games, games_count of them being played in parallel.

        IMPORTANT: Changing games_count also requires to change global variable
        BATCH_SIZE to preserve same ratio of forward_moves / learning_moves ratio.
        Games_count equal 16 was tested to be optimal for speed on my device.

        """
        main_net, target_net, storage = self._get_components()
        optimizer = self._get_optimizer(main_net, LEARING_OPTIM_RATE)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=STEP_TO_MODIFY, gamma=GAMMA
        )
        batch_start_time = time.perf_counter()
        eps = EPS_MAX

        states = bl.VectorizedStates(games_count)
        games_played = 0
        games_played_mod = 0
        loops = 0
        while games_played < games_to_play:
            # GATHER MOVES AND CHOOSE ONE PER GAME
            _, R, T, B, L, S, MC = self._gather_data(states)
            q_per_move = self._forward_data(R, T, B, L, S, main_net)
            rows = self._agent_decide(q_per_move, MC, eps)

            # PLAY THEM
            term_sum = self._env_update(
                states, storage, R[rows], T[rows], B[rows], L[rows], S[rows]
            )

            self._perform_gradient_descent(
                main_net, target_net, storage, optimizer, loops % DISPLAY_ERR_FREQ == 0
            )

            loops += 1
            games_played += term_sum
            games_played_mod += term_sum
            for _ in range(term_sum):
                scheduler.step()
            eps = max(EPS_MIN, eps * (DECAY_RATE**games_count))
            games_played_mod, batch_start_time = self._display_info(
                games_played, games_played_mod, batch_start_time, eps, main_net
            )

            # EXCHANGE NETWORKS
            if loops % EXCHANGE_NETS_FREQ == 0:
                target_net.load_state_dict(main_net.state_dict())


class AdvancedTrainer(StateNetworkTrainer):
    """Class for working with auxiliary training of BL_DQN_State_Network_v2 model,
    with default - medium size.
    """

    SAVE_PATH_NET = NETWORKS_DIR / "state_network_v2.pt"
    SAVE_PATH_SCORE = NETWORKS_DIR / "best_score_v2.txt"
    NUM_AUX = 5

    def _get_components(self):  # type: ignore
        # fmt: off
        return (
            net.BL_DQN_State_Network_v2(net.BL_DQN_State_Network(), self.NUM_AUX).to(bl.DEVICE),
            net.BL_DQN_State_Network_v2(net.BL_DQN_State_Network(), self.NUM_AUX).to(bl.DEVICE),
            net.SmartStateStorage(),
        )
        # fmt: on

    def _perform_gradient_descent(
        self, main_net, target_net, storage, optimizer, display_err: bool = False
    ):
        if storage.size < BATCH_SIZE:
            return

        # fmt: off
        (in_brd_s, in_blk_s, in_str_s, 
         in_brd_f, in_blk_f, in_str_f,
         b_rewards, b_terminals) = storage.sample(BATCH_SIZE)
        # fmt: on

        b_terminals = b_terminals.float()
        b_rewards = b_rewards.float()
        nn_brd_s = in_brd_s.float()

        v_pred, aux_pred = main_net(nn_brd_s, in_blk_s.long(), in_str_s.float())
        v_pred = v_pred.squeeze(1)

        with torch.no_grad():
            v_next, _ = target_net(in_brd_f.float(), in_blk_f.long(), in_str_f.float())
            v_next = v_next.squeeze(1)

        target_v = b_rewards + (1 - b_terminals) * DISCOUNT * v_next
        loss_v = F.smooth_l1_loss(v_pred, target_v)

        target_aux = calculate_aux_targets(nn_brd_s)
        loss_aux = F.mse_loss(aux_pred, target_aux)

        total_loss = loss_v + SIDE_TASK_WEIGHT * loss_aux
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


class LightTrainer(StateNetworkTrainer):
    """Class for working with BL_DQN_State_Network_v1() light version model."""

    SAVE_PATH_NET = NETWORKS_DIR / "state_network_light.pt"
    SAVE_PATH_SCORE = NETWORKS_DIR / "best_score_light.txt"

    def _get_components(self):  # type: ignore
        # fmt: off
        return (
            net.BL_DQN_State_Network_v1(net.BL_DQN_State_Network_Light()).to(bl.DEVICE),
            net.BL_DQN_State_Network_v1(net.BL_DQN_State_Network_Light()).to(bl.DEVICE),
            net.StateStorage(),
        )
        # fmt: on

class HeavyTrainer(StateNetworkTrainer):
    """Class for working with BL_DQN_State_Network_v1() heavy version model."""

    SAVE_PATH_NET = NETWORKS_DIR / "state_network_heavy.pt"
    SAVE_PATH_SCORE = NETWORKS_DIR / "best_score_heavy.txt"

    def _get_components(self):  # type: ignore
        # fmt: off
        return (
            net.BL_DQN_State_Network_v1(net.BL_DQN_State_Network_Heavy()).to(bl.DEVICE),
            net.BL_DQN_State_Network_v1(net.BL_DQN_State_Network_Heavy()).to(bl.DEVICE),
            net.StateStorage(),
        )
        # fmt: on

    def _get_optimizer(self, net: nn.Module, lr: float, weight_decay: float = 1e-4):
        decay, no_decay = [], []
        for name, p in net.named_parameters():
            if p.ndim <= 1 or "embed" in name:  # biasy (1D) i embeddingi
                no_decay.append(p)
            else:
                decay.append(p)
        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=lr,
        )

class TimedMixin:
    """Helper class to run diagnostics on what parts of code are the slowest.

    Mix with one of the trainers, run code inside `with trainer.timed_run():`,
    then call trainer.profiler.report(). Functions listed below get timed,
    calls nested inside other timed ones show up as their subsections.
    """

    # fmt: off
    STAGES = {"_gather_data": "gather", "_forward_data": "forward",
              "_agent_decide": "agent_decision", "_env_update": "env_update",
              "_play_parallel_games": "play", "_perform_gradient_descent": "backprop",
              "_display_info": "display_info"}
    STORAGE = {"add_records": "storage.add_records", "sample": "storage.sample"}
    PARTS = [
        (bl.VectorizedStates, {name: name.strip("_") for name in (
            "states_prim", "legal_moves", "_available_after_moves", "_simulate_transitions",
            "_has_legal_move", "mass_transition", "reset", "drop")}),
        (net.StateStorage, STORAGE), (net.SmartStateStorage, STORAGE), (net.ActionsStorage, STORAGE),
        (sys.modules[__name__], {"_segment_argmax": "segment_argmax", "val_net": "val_net",
                                 "try_write_best_net": "try_write_best_net"}),
    ]
    # fmt: on

    def __init__(self) -> None:
        super().__init__()
        self.profiler = profiling.Profiler()

    @contextmanager
    def timed_run(self, name: str = "run"):
        """Time everything inside the `with` block, listed functions become subsections."""
        with self.profiler.instrument([(self, self.STAGES), *self.PARTS]), self.profiler.section(name):
            yield self.profiler


# fmt: off
class T_StateNetworkTrainer(TimedMixin, StateNetworkTrainer):
    pass
class T_AdvancedTrainer(TimedMixin, AdvancedTrainer):
    pass
class T_LightTrainer(TimedMixin, LightTrainer):
    pass
class T_HeavyTrainer(TimedMixin, HeavyTrainer):
    pass
class T_ActionNetworkTrainer(TimedMixin, ActionNetworkTrainer):
    pass
# fmt: on


def _train_fn(trainer):
    if isinstance(trainer, ActionNetworkTrainer):
        return trainer.train_action_network
    return trainer.train_state_network

def main(args):
    torch.manual_seed(bl.DEF_SEED)
    if len(args) < 2:
        print("Please provide a command! Use command 'usage' for help!")
        return 0

    if args[1] == "usage":
        print(
            """
    Calling convention: <{python3 / uv run / python}> <execpath> <command> <model (optional)> <games (optional)>

    games: number of games to play, defaults are 60000 (train), 500 (test), 3000 (time_profile).

    Commands available for blokudoku_AI:
    train: trains the AI model
    test: tests the AI model
    play_as_player: play the game as a player, to compare score with the AI
    watch_ai_play: AI plays slowly, so you can watch it play
    time_profile: Performance measurements towards optimalization

    Models available for training and testing:
    casual: BL_DQN_State_Network_v1 model (BEST RESULTS by far, recommended)
    advanced: BL_DQN_State_Network_v2 model with auxiliary tasks

    light: BL_DQN_State_Network_Light model (underfitting, not recommended)
    heavy: BL_DQN_State_Network_Heavy model (very overfitting, not recommended)
    action: BL_DQN_Action_Network_v1 model (very bad results, not recommended)

    Networks and their best scores are saved under networks/. To alter
    filenames, change SAVE_PATH_NET, SAVE_PATH_SCORE inside classes of main.py.
    """
        )
        return 0
    if args[1] == "play_as_player":
        bl.rng_fact_instance = bl.RandomFactory()
        user_play.play_as_player()
        return 0

    if args[1] not in ["train", "test", "watch_ai_play", "time_profile"]:
        print(f"Unrecognised argument: '{args[1]}'! Use command 'usage' for help! ")
        return 0
    if len(args) not in (3, 4):
        print(
            f"Please provide a command and a model name! Use command 'usage' for help!"
        )
        return 0

    default_games = {
        "train": GAMES_TO_PLAY_TRAIN,
        "test": GAMES_TO_PLAY_TEST,
        "time_profile": GAMES_TO_PLAY_PROFILE,
    }
    games_to_play = default_games.get(args[1], 0)
    if len(args) == 4:
        try:
            games_to_play = int(args[3])
        except ValueError:
            games_to_play = 0
        if games_to_play <= 0:
            print(f"Number of games must be a positive integer, got: '{args[3]}'!")
            return 0

    models = {
        "casual": StateNetworkTrainer(),
        "advanced": AdvancedTrainer(),
        "light": LightTrainer(),
        "heavy": HeavyTrainer(),
        "action": ActionNetworkTrainer(),
    }
    if args[2] not in models.keys():
        print(f"Unrecognised model: '{args[2]}'! Use command 'usage' for help! ")
        return 0
    trainer = models[args[2]]

    if args[1] == "train":
        _train_fn(trainer)(games_to_play)
    elif args[1] == "test":
        trainer.test_network(games_to_play)
    elif args[1] == "watch_ai_play":
        trainer.showcase_network()
    elif args[1] == "time_profile":
        timed_models = {
            "casual": T_StateNetworkTrainer,
            "advanced": T_AdvancedTrainer,
            "light": T_LightTrainer,
            "heavy": T_HeavyTrainer,
            "action": T_ActionNetworkTrainer,
        }
        trainer = timed_models[args[2]]()
        print(f"Beginning diagnostics on {games_to_play} games of training!")

        with trainer.timed_run("train"):
            _train_fn(trainer)(games_to_play)
        trainer.profiler.report()


def cli():
    """Entry point of the `blokudoku` command (see [project.scripts])."""
    main(sys.argv)

if __name__ == "__main__":
    cli()
