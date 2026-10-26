import blokudoku_lib as bl
import network_lib as net
import user_play as user_play

import numpy as np
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# LONGEVITY OF TRAINING AND VALIDATION INTERVAL PARAMETERS
GAMES_TO_PLAY = 60_000
GAMES_IN_PARALLEL = 16
EXCHANGE_NETS_FREQ = 250
DISPLAY_ERR_FREQ = 1500
GAMES_TO_DISPLAY_INFO = 250
VALIDATION_GAMES_PER_BATCH = 5
VALIDATION_GAMES_PER_FIGHT = 25

# BASIC LEARING HYPERPARAMETERS
EPS_MAX = 1.0
EPS_MIN = 0.005
DECAY_RATE = 0.9999975
DISCOUNT = 0.995
BATCH_SIZE = 128
LEARING_OPTIM_RATE = 1e-4

# ADVANCED OPTIONS
MARGIN = 50.0
SIDE_TASK_WEIGHT = 7.0
EPS_THRESHOLD = 0.15
STEP_TO_MODIFY = 15000
GAMMA = 0.5

def try_write_best_net(netw: nn.Module, 
                       filepath_net: str, 
                       filepath_score: str, 
                       val_games_num: int,
                       netw_res: float = 0.0):
    """ Save network and results it achieved.

    Used as validation tool in model selection. Be cautious to invoke it
    only when same type of network is scored in file filepath_net, otherwise
    may result in fatal error. Follows algorithm below:

    1. If there is no proper save of network under filepath_net & filepath_score,
    save netw and netw_res (return)

    2. Otherwise, compare netw_res with score on filepath_score, if netw_res is 
    worse more than MARGIN points, forfeit saving to not overwrite better model.

    3. If scores are similar, perform additional validation on val_games_num
    games to save better performing net.

    Args:
        netw: network model to save
        filepath_net: filepath to save netw
        filepath_score: filepath to save netw result
        val_games_num: number of validation games to perform, if needed.
        netw_res: result network achieved when validated first.
    """
    if not os.path.exists(filepath_score) or not os.path.exists(filepath_net):
        torch.save(netw.state_dict(), filepath_net)
        current_score, _ = val_state_net(netw, val_games_num) # type: ignore
        
        with open(filepath_score, "w") as f:
            f.write(str(current_score)) 
        print(f"Base score set: {current_score:.2f}")
        return

    with open(filepath_score, "r") as f:
        try:
            saved_score = float(f.read())
        except ValueError:
            saved_score = -1e2
    if netw_res + MARGIN < saved_score:
        return

    saved_net = copy.deepcopy(netw)
    load_success = False
    try:
        saved_net.load_state_dict(torch.load(filepath_net, weights_only=True))
        load_success = True
    except Exception as e:
        load_success = False

    if not load_success:
        torch.save(netw.state_dict(), filepath_net)
        current_score, _ = val_state_net(netw, val_games_num) # type: ignore
        
        with open(filepath_score, "w") as f:
            f.write(str(current_score))
        print(f"New best score: {current_score:.2f}")
        return

    print(f"\nStarting game validation - two agent fight ...")
    current_score, _ = val_state_net(netw, val_games_num) # type: ignore
    saved_score, _ = val_state_net(saved_net, val_games_num) # type: ignore

    # Kto wygrał?
    if current_score > saved_score:
        print(f"We got new champion! - result of: {current_score:.2f} vs {saved_score:.2f}")
        torch.save(netw.state_dict(), filepath_net)
        with open(filepath_score, "w") as f:
            f.write(str(current_score))
    else:
        print(f"Saved net held - result of: {current_score:.2f} vs {saved_score:.2f}")
        with open(filepath_score, "w") as f:
            f.write(str(saved_score))

def val_state_net(netw: nn.Module, games_count: int = 100, more_info: bool = False):
    """ Function that validates netw performance on games_count games.

    Args:
        games_count: games to play
        more_info: whether to return more

    Returns:
        mean: mean of achieved results
        std: std of achieved results
        max (only if more_info=True): max of achieved results
        min (only if more_info=True): min of achieved results
    """
    scores = []
    netw.eval()

    for _ in range(games_count):
        cur_state = bl.State()
        mask = cur_state.calculate_mask()
        game_over = False
        score = 0
        while not game_over:
            actions, rewards, _, boards, blocks, streaks = cur_state.all_1move_states(mask)

            nn_board = torch.from_numpy(boards).float()
            nn_blocks = torch.from_numpy(blocks).long()
            nn_streak = torch.from_numpy(streaks).float().view(-1, 1)
            
            with torch.no_grad():
                out = netw(nn_board, nn_blocks, nn_streak)
                v_values = out[0].squeeze(-1) if isinstance(out, tuple) else out.squeeze(-1)
                y_hat_nn = v_values + torch.from_numpy(rewards).float()

            best_idx = int(torch.argmax(y_hat_nn))
            best_action_encoded = actions[best_idx]
            
            piece, x, y = bl.State.decode_action(best_action_encoded)
            rwrd, game_over, mask = cur_state.transition(piece, x, y)
            score += int(rwrd)

        scores.append(score - bl.NEURAL_PENALTY)
    netw.train()
    if more_info:
        res = np.mean(scores), np.std(scores), np.max(scores), np.min(scores)
    else:
        res = np.mean(scores), np.std(scores)
    return res

def calculate_aux_targets(boards_tensor: torch.Tensor):
    """ Calculate auxiliray targets for a boards_tensor,
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
    f4 = torch.stack([
        b[:, i:i+3, j:j+3].sum(dim=(1, 2)) > 0
        for i in range(0, 9, 3) for j in range(0, 9, 3)
    ], dim=1).sum(dim=1)
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

def _randomise(mask: np.ndarray):
    valid_actions = np.flatnonzero(mask)
    return bl.rng_fact_instance.choice(valid_actions)

class ActionNetworkTrainer:
    """ Class capable of triaing and testing BL_DQN_Action_Network_v1().

    Defaultly saves network under path ActionNetworkTrainer.SAVE_PATH, 
    feel free to overwrite if different path is desired.
    """
    SAVE_PATH = "action_network_v1.pt"

    @staticmethod
    def test_network():
        """Tests netowork on 100 games.
        """
        main_net = net.BL_DQN_Action_Network_v1()

        try:
            main_net.load_state_dict(torch.load(ActionNetworkTrainer.SAVE_PATH, weights_only=True))
        except Exception as e:
            print(f"Error - didn't managed to find network params in filepath{ActionNetworkTrainer.SAVE_PATH}\n")
            return

        main_net.validate_model(100)



    @staticmethod
    def _play_parallel_games(states: list, 
                             main_net: net.BL_DQN_Action_Network_v1,
                             storage: net.ActionsStorage,
                             eps: float, 
                             games_count: int):
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
        decoded_actions = [bl.State.decode_action(actions[i]) for i in range(games_count)]

        rewards = np.empty(games_count, dtype=np.int16)
        terminals = np.empty(games_count, dtype=bool)
        new_masks = np.empty((games_count, bl.NN_OUTPUT_FLAT), dtype=bool)
        for i in range(games_count):
            rewards[i], terminals[i], new_masks[i] = states[i].transition(*decoded_actions[i])

        storage.add_records(
            np.array(boards), np.array(blocks), np.array(streaks),
            np.array([st.in_board() for st in states]),
            np.array([st.in_blocks() for st in states]),
            np.array([st.in_streak() for st in states]),
            np.array(actions), rewards, terminals, new_masks
        )
        return terminals

    @staticmethod
    def _perform_gradient_descent(main_net: net.BL_DQN_Action_Network_v1, 
                                  target_net: net.BL_DQN_Action_Network_v1,
                                  storage: net.ActionsStorage, 
                                  optimizer: torch.optim.Adam):
        if storage.size < BATCH_SIZE:
            return

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

    @staticmethod
    def _exchange_networks(loops: int, 
                           main_net: net.BL_DQN_Action_Network_v1, 
                           target_net: net.BL_DQN_Action_Network_v1):
        if loops % EXCHANGE_NETS_FREQ == 0:
            target_net.load_state_dict(main_net.state_dict())

    @staticmethod
    def _display_info(games_played: int, 
                      games_played_mod: int, 
                      batch_start_time: float, 
                      eps: float, 
                      main_net: net.BL_DQN_Action_Network_v1):

        if games_played_mod < GAMES_TO_DISPLAY_INFO:
            return games_played_mod, batch_start_time

        elapsed = time.perf_counter() - batch_start_time
        games_per_second = GAMES_TO_DISPLAY_INFO / elapsed if elapsed else float("inf")

        print(f"Games played: {games_played}, Games per second: " +
              f"{games_per_second:.2f}, batch time: {elapsed:.2f}s, eps: {eps:.4f}")

        print(f"Validating model performance...")
        print(main_net.validate_model(VALIDATION_GAMES_PER_BATCH))
        print("\n")

        return games_played_mod % GAMES_TO_DISPLAY_INFO, time.perf_counter()

    @staticmethod
    def train_action_network(games_count: int = GAMES_IN_PARALLEL):
        """ Trains network on games_count games being played in parallel. 
        
        IMPORTANT: Changing games_count also requires to change global variable 
        BATCH_SIZE to preserve same ratio of forward_moves / learning_moves ratio.
        Games_count equal 16 was tested to be optimal for speed on my device.

        """
        main_net = net.BL_DQN_Action_Network_v1()
        target_net = net.BL_DQN_Action_Network_v1()
        storage = net.ActionsStorage()

        optimizer = torch.optim.Adam(main_net.parameters(), lr=LEARING_OPTIM_RATE)
        batch_start_time = time.perf_counter()
        eps = EPS_MAX

        states = [bl.State() for _ in range(games_count)]
        games_played = 0
        games_played_mod = 0
        loops = 0
        while games_played < GAMES_TO_PLAY:
            terminals = ActionNetworkTrainer._play_parallel_games(
                states, main_net, storage, eps, games_count
            )
            ActionNetworkTrainer._perform_gradient_descent(
                main_net, target_net, storage, optimizer
            )

            loops += 1
            eps = max(EPS_MIN, eps * (DECAY_RATE ** games_count))
            games_played += np.sum(terminals)
            games_played_mod += np.sum(terminals)
            for i in range(games_count):
                if terminals[i]:
                    states[i] = bl.State()
    
            ActionNetworkTrainer._exchange_networks(loops, main_net, target_net)
            games_played_mod, batch_start_time = ActionNetworkTrainer._display_info(
                games_played, games_played_mod, batch_start_time, eps, main_net
            )

        torch.save(main_net.state_dict(), ActionNetworkTrainer.SAVE_PATH)

class StateNetworkTrainer:
    """ Class for working with default BL_DQN_State_Network_v1() NN model.
    """
    SAVE_PATH_NET = "state_network_v1.pt"
    SAVE_PATH_SCORE = "best_score.txt"

    @classmethod
    def showcase_network(cls):
        """ Interactive display in terminal (stdout), of how network saved under
        SAVE_PATH_NET plays the game.
        """
        main_net, _, _ = cls._get_components()
        try:
            main_net.load_state_dict(torch.load(cls.SAVE_PATH_NET, weights_only=True))
        except Exception as e:
            print(f"Error - didn't managed to find network params in filepath{cls.SAVE_PATH_NET}\n")
            return

        main_net.eval()
        score = 0
        game_over = False
        state = bl.State()
        while not game_over:
            user_play.output_to_player(state, score, game_over)
            (a_actions, a_rewards, a_terminals, a_boards, 
             a_blocks, a_streaks, lengths) = cls._gather_data([state], [state.calculate_mask()])

            q_per_game = cls._forward_data(a_rewards, a_terminals, a_boards,
                                                a_blocks, a_streaks, main_net, lengths)
            chosen_actions = cls._agent_decide(q_per_game, a_actions, 0.00)
            rwrd, game_over, _ = state.transition(*bl.State.decode_action(a_actions[0][chosen_actions[0]]))
            score += rwrd
            time.sleep(2)

        user_play.output_to_player(state, score, game_over)

    @classmethod
    def test_network(cls, games_count: int = 100):
        """ Validate network on the set of games_count games. Print result to stdout.
        """
        main_net, _, _ = cls._get_components()
        try:
            main_net.load_state_dict(torch.load(cls.SAVE_PATH_NET, weights_only=True))
        except Exception as e:
            print(f"Error - didn't managed to find network params in filepath{cls.SAVE_PATH_NET}\n")
            return

        score, score_std, max, min = val_state_net(main_net, games_count, True) # type: ignore
        print(f"Tested model performance - mean score: {score:.2f}, std: {score_std:.2f}, \
max:{max:.4f}, min:{min:.4f}")


    @classmethod
    def _get_components(cls):
        return net.BL_DQN_State_Network_v1(net.BL_DQN_State_Network()), \
               net.BL_DQN_State_Network_v1(net.BL_DQN_State_Network()), \
               net.StateStorage()

    @classmethod
    def _gather_data(cls, states: list, masks: list):
        (all_actions, all_rewards, all_terminals, 
         all_boards, all_blocks, all_streaks)  = [], [], [], [], [], []
        lengths = []

        for i in range(len(masks)):
            actions, rewards, terminals, boards, blocks, streaks = states[i].all_1move_states(masks[i])
            all_actions.append(actions)
            all_rewards.append(rewards)
            all_terminals.append(terminals)
            all_boards.append(boards)
            all_blocks.append(blocks)
            all_streaks.append(streaks)
            lengths.append(len(actions))
        return all_actions, all_rewards, all_terminals, all_boards, all_blocks, all_streaks, lengths

    @classmethod
    def _forward_data(cls, rewards: list, terminals: list, boards: list, blocks: list, streaks: list,
                      nn_net: nn.Module, lengths: list):
        mega_rewards = torch.from_numpy(np.concatenate(rewards)).float()
        mega_terminals = torch.from_numpy(np.concatenate(terminals)).float()
        mega_nn_boards = torch.from_numpy(np.concatenate(boards)).float()
        mega_nn_blocks = torch.from_numpy(np.concatenate(blocks)).long()
        mega_nn_streaks = torch.from_numpy(np.concatenate(streaks)).float()

        with torch.no_grad():
            out = nn_net(mega_nn_boards, mega_nn_blocks, mega_nn_streaks)
            v_values = out[0].squeeze(1) if isinstance(out, tuple) else out.squeeze(1)

        q_values = mega_rewards + (1 - mega_terminals) * DISCOUNT * v_values
        return torch.split(q_values, lengths)

    @classmethod
    def _agent_decide(cls,qs: tuple, act: list, eps: float):
        n = len(qs)
        choices = np.empty(n, dtype=np.int16)
        explorations = [bl.rng_fact_instance.explore_now(eps) for _ in range(n)]
        for i in range(n):
            if explorations[i]:
                choices[i] = bl.rng_fact_instance.choice(range(len(act[i])))
            else:
                choices[i] = int(torch.argmax(qs[i]))
        return choices

    @classmethod
    def _env_update(cls, states: list, masks: list, chosen_actions: np.ndarray, 
                    storage, all_actions: list):
        n = chosen_actions.shape[0]
        boards_s = np.array([st.in_board() for st in states])
        blocks_s = np.array([st.in_blocks() for st in states])
        streaks_s = np.array([st.in_streak() for st in states])

        rewards = np.empty(n, dtype=np.int16)
        terminals = np.empty(n, dtype=bool)
        for i in range(n):
            action_code = all_actions[i][chosen_actions[i]]
            piece, x, y = bl.State.decode_action(action_code)
            rewards[i], terminals[i], masks[i] = states[i].transition(piece, x, y)

        boards_f = np.array([st.in_board() for st in states])
        blocks_f = np.array([st.in_blocks() for st in states])
        streaks_f = np.array([st.in_streak() for st in states])

        storage.add_records(
            boards_s, blocks_s, streaks_s,
            boards_f, blocks_f, streaks_f,
            rewards, terminals
        )

        for i in range(n):
            if terminals[i]:
                states[i] = bl.State()
                masks[i] = states[i].calculate_mask()      
        return states, masks, np.sum(terminals)

    @classmethod
    def _perform_gradient_descent(cls, main_net: nn.Module, 
                                  target_net: nn.Module,
                                  storage,
                                  optimizer: torch.optim.Adam, display_err: bool = False):
        if storage.size < BATCH_SIZE:
            return

        (in_brd_st1, in_blk_st1, in_str_st1,
            in_brd_st2, in_blk_st2, in_str_st2,
            b_rewards, b_terminals) = storage.sample(BATCH_SIZE)
        b_terminals = torch.from_numpy(b_terminals).float()
        b_rewards = torch.from_numpy(b_rewards).float()

        y_hat_nn_main = main_net(
            torch.from_numpy(in_brd_st1).float(),
            torch.from_numpy(in_blk_st1).long(),
            torch.from_numpy(in_str_st1).float()
        ).squeeze(1)
        with torch.no_grad():
            y_hat_nn_target = target_net(
                torch.from_numpy(in_brd_st2).float(),
                torch.from_numpy(in_blk_st2).long(),
                torch.from_numpy(in_str_st2).float()
            ).squeeze(1)

        target_fn = b_rewards + (1 - b_terminals) * DISCOUNT * y_hat_nn_target
        losses = F.mse_loss(y_hat_nn_main, target_fn)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    @classmethod
    def _display_info(cls, games: int, games_mod: int,
                           start: float, eps: float,
                           main_net: nn.Module):
        if games_mod < GAMES_TO_DISPLAY_INFO or games == 0:
            return games_mod, start

        elapsed = time.perf_counter() - start
        games_per_second = GAMES_TO_DISPLAY_INFO / elapsed if elapsed else float("inf")
        print(f"Games played: {games}, Games per second: " +
                f"{games_per_second:.2f}, batch time: {elapsed:.2f}s, eps: {eps:.4f}")
        print(f"Validating model performance...")

        score, score_std = val_state_net(main_net, VALIDATION_GAMES_PER_BATCH) # type: ignore
        print(f"Points in val games - mean: {score:.2f}, std: {score_std:.2f}")
        print()

        try_write_best_net(main_net, cls.SAVE_PATH_NET, 
            cls.SAVE_PATH_SCORE, VALIDATION_GAMES_PER_FIGHT, score)
        return games_mod % GAMES_TO_DISPLAY_INFO, time.perf_counter()



    @classmethod
    def train_state_network(cls, games_count: int = GAMES_IN_PARALLEL):
        """ Trains network on games_count games being played in parallel. 
        
        IMPORTANT: Changing games_count also requires to change global variable 
        BATCH_SIZE to preserve same ratio of forward_moves / learning_moves ratio.
        Games_count equal 16 was tested to be optimal for speed on my device.

        """
        main_net, target_net, storage = cls._get_components()
        optimizer = torch.optim.Adam(main_net.parameters(), lr=LEARING_OPTIM_RATE)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_TO_MODIFY, gamma=GAMMA)
        batch_start_time = time.perf_counter()
        eps = EPS_MAX

        states = [bl.State() for _ in range(games_count)]
        masks = [st.calculate_mask() for st in states]
        games_played = 0
        games_played_mod = 0
        loops = 0
        while games_played < GAMES_TO_PLAY:
            # GATHER MOVES
            (a_actions, a_rewards, a_terminals, a_boards, 
             a_blocks, a_streaks, lengths) = cls._gather_data(states, masks)

            q_per_game = cls._forward_data(a_rewards, a_terminals, a_boards,
                                                a_blocks, a_streaks, main_net, lengths)
            chosen_actions = cls._agent_decide(q_per_game, a_actions, eps)

            states, masks, term_sum = cls._env_update(states, masks, 
                    chosen_actions, storage, a_actions)

            cls._perform_gradient_descent(main_net, target_net, storage, 
                                          optimizer, loops % DISPLAY_ERR_FREQ == 0)

            loops += 1
            games_played += term_sum
            games_played_mod += term_sum
            for _ in range(term_sum): 
                scheduler.step()
            eps = max(EPS_MIN, eps * (DECAY_RATE ** games_count))
            games_played_mod, batch_start_time = cls._display_info(
                games_played, games_played_mod, batch_start_time, eps, main_net
            )

            # EXCHANGE NETWORKS
            if loops % EXCHANGE_NETS_FREQ == 0:
                target_net.load_state_dict(main_net.state_dict())

class AdvancedTrainer(StateNetworkTrainer):
    """ Class for working with auxiliary training of BL_DQN_State_Network_v2 model,
    with default - medium size.
    """
    SAVE_PATH_NET = "state_network_v2.pt"
    SAVE_PATH_SCORE = "best_score_v2.txt"
    NUM_AUX = 5

    @classmethod
    def _get_components(cls): # type: ignore
        return net.BL_DQN_State_Network_v2(net.BL_DQN_State_Network(), cls.NUM_AUX), \
               net.BL_DQN_State_Network_v2(net.BL_DQN_State_Network(), cls.NUM_AUX), \
               net.SmartStateStorage()
    
    @classmethod
    def _perform_gradient_descent(cls, main_net, target_net, storage, optimizer, display_err: bool = False):
        if storage.size < BATCH_SIZE:
            return

        (in_brd_s, in_blk_s, in_str_s,
         in_brd_f, in_blk_f, in_str_f,
         b_rewards, b_terminals) = storage.sample(BATCH_SIZE)
         
        b_terminals = torch.from_numpy(b_terminals).float()
        b_rewards = torch.from_numpy(b_rewards).float()
        nn_brd_s = torch.from_numpy(in_brd_s).float()

        v_pred, aux_pred = main_net(
            nn_brd_s, 
            torch.from_numpy(in_blk_s).long(), 
            torch.from_numpy(in_str_s).float()
        )
        v_pred = v_pred.squeeze(1)

        with torch.no_grad():
            v_next, _ = target_net(
                torch.from_numpy(in_brd_f).float(),
                torch.from_numpy(in_blk_f).long(),
                torch.from_numpy(in_str_f).float()
            )
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
    """ Class for working with BL_DQN_State_Network_v1() light version model.
    """
    SAVE_PATH_NET = "state_network_light.pt"
    SAVE_PATH_SCORE = "best_score_light.txt"

    @classmethod
    def _get_components(cls): # type: ignore
        return net.BL_DQN_State_Network_v1(net.BL_DQN_State_Network_Light()), \
               net.BL_DQN_State_Network_v1(net.BL_DQN_State_Network_Light()), \
               net.StateStorage()

class HeavyTrainer(StateNetworkTrainer):
    """ Class for working with BL_DQN_State_Network_v1() heavy version model.
    """
    SAVE_PATH_NET = "state_network_heavy.pt"
    SAVE_PATH_SCORE = "best_score_heavy.txt"

    @classmethod
    def _get_components(cls): # type: ignore
        return net.BL_DQN_State_Network_v1(net.BL_DQN_State_Network_Heavy()), \
               net.BL_DQN_State_Network_v1(net.BL_DQN_State_Network_Heavy()), \
               net.StateStorage()

# TODO - add options to switch between models for user, via console.
def main(args):
    if len(args) < 2:
        print("Please provide a command! Use command 'usage' for help!")
        return 0

    if args[1] == "usage":
        print(
    """Commands available for blokudoku_AI:
    train: trains the AI model
    test: tests the AI model
    play_as_player: play the game as a player, to compare score with the AI
    watch_ai_play: AI plays slowly, so you can watch it play

    Models available for training and testing:
    casual: BL_DQN_State_Network_v1 model (BEST RESULTS by far, recommended)
    advanced: BL_DQN_State_Network_v2 model with auxiliary tasks

    light: BL_DQN_State_Network_Light model (underfitting, not recommended)
    heavy: BL_DQN_State_Network_Heavy model (very overfitting, not recommended)
    action: BL_DQN_Action_Network_v1 model (very bad results, not recommended)
    """)
    elif args[1] == "play_as_player":
        bl.rng_fact_instance = bl.RandomFactory()
        user_play.play_as_player()
    elif len(args) != 3:
        print(f"Please provide a command and a model name! Use command 'usage' for help!")
        return 0

    if args[1] == "train":
        if args[2] == "casual":
            StateNetworkTrainer.train_state_network()
        elif args[2] == "advanced":
            AdvancedTrainer.train_state_network()
        elif args[2] == "light":
            LightTrainer.train_state_network()
        elif args[2] == "heavy":
            HeavyTrainer.train_state_network()
        elif args[2] == "action":
            ActionNetworkTrainer.train_action_network()

    elif args[1] == "test":
        if args[2] == "casual":
            StateNetworkTrainer.test_network()
        elif args[2] == "advanced":
            AdvancedTrainer.test_network()
        elif args[2] == "light":
            LightTrainer.test_network()
        elif args[2] == "heavy":
            HeavyTrainer.test_network()
        elif args[2] == "action":
            ActionNetworkTrainer.test_network()

    elif args[1] == "watch_ai_play":
        bl.rng_fact_instance = bl.RandomFactory()
        if args[2] == "casual":
            StateNetworkTrainer.showcase_network()
        elif args[2] == "advanced":
            AdvancedTrainer.showcase_network()
        elif args[2] == "light":
            LightTrainer.showcase_network()
        elif args[2] == "heavy":
            HeavyTrainer.showcase_network()
        elif args[2] == "action":
            raise RuntimeError("Not implemented yet!")
            ActionNetworkTrainer.train_action_network()
        pass

    else:
        print(f"Unrecgonised argument: {args[1]}! Use command 'usage' for help! ")
        return 0


if __name__ == "__main__":
    main(sys.argv)
