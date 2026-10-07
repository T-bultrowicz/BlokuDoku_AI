import blokudoku_lib as bl
import network_lib as net
import user_play as user_play

import numpy as np
import sys
import time
import torch
import torch.nn.functional as F

GAMES_TO_PLAY = 100_000
GAMES_IN_PARALLEL = 16

EXCHANGE_NETS_FREQ = 250
GAMES_TO_DISPLAY_INFO = 1000

VALIDATION_GAMES_PER_BATCH = 7
EPS_MAX = 1.0
EPS_MIN = 0.025
DECAY_RATE = 0.999995
DISCOUNT = 0.999
BATCH_SIZE = 128
LEARING_OPTIM_RATE = 1e-4

def _randomise(mask: np.ndarray):
    valid_actions = np.flatnonzero(mask)
    return bl.rng_fact_instance.choice(valid_actions)

class ActionNetworkTrainer:
    SAVE_PATH = "action_network_v1.pt"

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
    SAVE_PATH = "state_network_v1.pt"

    @staticmethod
    def train_state_network(games_count: int = GAMES_IN_PARALLEL):
        main_net = net.BL_DQN_State_Network_v1()
        target_net = net.BL_DQN_State_Network_v1()
        storage = net.StateStorage()

        optimizer = torch.optim.Adam(main_net.parameters(), lr=LEARING_OPTIM_RATE)
        batch_start_time = time.perf_counter()
        eps = EPS_MAX

        states = [bl.State() for _ in range(games_count)]
        masks = [st.calculate_mask() for st in states]
        games_played = 0
        games_played_mod = 0
        loops = 0
        while games_played < GAMES_TO_PLAY:
            # GATHER MOVES
            all_actions, all_rewards, all_boards, all_blocks, all_streaks  = [], [], [], [], []
            lengths = []
            for i in range(games_count):
                actions, rewards, boards, blocks, streaks = states[i].all_1move_states(masks[i])
                all_actions.append(actions)
                all_rewards.append(rewards)
                all_boards.append(boards)
                all_blocks.append(blocks)
                all_streaks.append(streaks)
                lengths.append(len(actions))

            # PERFORM GRADIENT DESCENT
            mega_nn_boards = torch.from_numpy(np.concatenate(all_boards)).float()
            mega_nn_blocks = torch.from_numpy(np.concatenate(all_blocks)).long()
            mega_nn_streaks = torch.from_numpy(np.concatenate(all_streaks)).float()
            mega_rewards = torch.from_numpy(np.concatenate(all_rewards)).float()
            with torch.no_grad():
                v_values = main_net(mega_nn_boards, mega_nn_blocks, mega_nn_streaks).squeeze(1)
            q_values = mega_rewards + DISCOUNT * v_values
            q_per_game = torch.split(q_values, lengths)

            # CHOOSE ACTIONS
            chosen_actions = np.empty(games_count, dtype=np.int16)
            explorations = [bl.rng_fact_instance.explore_now(eps) for _ in range(games_count)]
            for i in range(games_count):
                if explorations[i]:
                    chosen_actions[i] = bl.rng_fact_instance.choice(all_actions[i])
                else:
                    chosen_actions[i] = all_actions[i][int(torch.argmax(q_per_game[i]))]

            # PERFORM ACTIONS AND UPDATE STATES
            boards_s = np.array([st.in_board() for st in states])
            blocks_s = np.array([st.in_blocks() for st in states])
            streaks_s = np.array([st.in_streak() for st in states])
            terminals = np.empty(games_count, dtype=bool)
            rewards = np.empty(games_count, dtype=np.int16)

            for i in range(games_count):
                piece, x, y = bl.State.decode_action(chosen_actions[i])
                rewards[i], terminals[i], masks[i] = states[i].transition(piece, x, y)

            boards_f = np.array([st.in_board() for st in states])
            blocks_f = np.array([st.in_blocks() for st in states])
            streaks_f = np.array([st.in_streak() for st in states])

            storage.add_records(
                boards_s, blocks_s, streaks_s,
                boards_f, blocks_f, streaks_f,
                rewards, terminals
            )

            for i in range(games_count):
                if terminals[i]:
                    states[i] = bl.State()
                    masks[i] = states[i].calculate_mask()

            # PERFORM GRADIENT DESCENT
            if storage.size >= BATCH_SIZE:
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

            # DISPLAY INFO
            loops += 1
            games_played += np.sum(terminals)
            games_played_mod += np.sum(terminals)
            eps = max(EPS_MIN, eps * (DECAY_RATE ** games_count))

            if games_played_mod >= GAMES_TO_DISPLAY_INFO:
                elapsed = time.perf_counter() - batch_start_time
                games_per_second = GAMES_TO_DISPLAY_INFO / elapsed if elapsed else float("inf")
                print(f"Games played: {games_played}, Games per second: " +
                      f"{games_per_second:.2f}, batch time: {elapsed:.2f}s, eps: {eps:.4f}")
                print(f"Validating model performance...")
                print(main_net.validate_model(VALIDATION_GAMES_PER_BATCH))
                print("\n")
                games_played_mod %= GAMES_TO_DISPLAY_INFO

            # EXCHANGE NETWORKS
            if loops % EXCHANGE_NETS_FREQ == 0:
                target_net.load_state_dict(main_net.state_dict())

        torch.save(main_net.state_dict(), StateNetworkTrainer.SAVE_PATH)

                
def main(args):
    if len(args) != 2:
       print(f"Main needs one argument, but received: {len(args) - 1} arguments! Type './executable usage' for help!")
       return 0

    if args[1] == "train":
        # ActionNetworkTrainer.train_action_network()
        StateNetworkTrainer.train_state_network()
        # train_dual_network()
        pass

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
