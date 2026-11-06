import numpy as np
import pytest
import torch

import blokudoku_lib as bl
import main
import network_lib as net
from blokudoku_lib import BLOCKS_TO_PICK, BOARD_FLAT, BOARD_LEN, DEVICE, NEURAL_PENALTY

STATE_TRAINERS = [
    main.StateNetworkTrainer,
    main.AdvancedTrainer,
    main.LightTrainer,
    main.HeavyTrainer,
]


def small_net():
    return net.BL_DQN_State_Network_v1(net.BL_DQN_State_Network_Light()).to(DEVICE)


def small_components():
    return small_net(), small_net(), net.StateStorage(50_000)


def segments(lengths: torch.Tensor):
    """(start, stop) row range of every game inside stacked tensors."""
    stops = lengths.cumsum(0).tolist()
    return [(stop - n, stop) for stop, n in zip(stops, lengths.tolist())]


def reference_greedy_scores(netw, games: int):
    """Plain-loop greedy play, summed rewards with NEURAL_PENALTY included."""
    scores = torch.zeros(games, dtype=torch.int32, device=DEVICE)
    ids = torch.arange(games, device=DEVICE)
    vs = bl.VectorizedStates(games)
    while vs.N > 0:
        _, R, T, B, L, S, MC = vs.states_prim()
        with torch.no_grad():
            vals = netw(B.float(), L.long(), S.float()).view(-1)
        best = torch.tensor(
            [lo + int(vals[lo:hi].argmax()) for lo, hi in segments(MC)], device=DEVICE
        )
        scores[ids] += R[best]
        vs.mass_transition(B[best], L[best], S[best])
        vs.drop(T[best])
        ids = ids[~T[best]]
    return scores


def random_storage_records(n: int):
    """n records spread over all board crowdness buckets of SmartStateStorage."""
    fills = np.random.choice([0.1, 0.3, 0.5, 0.75], n)[:, None]
    boards_s = np.random.rand(n, BOARD_FLAT) < fills
    boards_f = np.random.rand(n, BOARD_FLAT) < fills
    blocks = np.random.randint(0, bl.BLOCKS_SIZE + 1, (n, BLOCKS_TO_PICK)).astype(np.uint8)
    streaks = np.random.rand(n, 1) < 0.5
    rewards = np.random.randint(-50, 40, n).astype(np.int16)
    terminals = np.random.rand(n) < 0.1
    # fmt: off
    return (boards_s, blocks, streaks,
            boards_f, blocks[::-1].copy(), streaks[::-1].copy(), rewards, terminals)
    # fmt: on


def params_snapshot(netw):
    return [p.detach().clone() for p in netw.parameters()]


class TestSegmentArgmax:
    def test_matches_per_game_argmax_with_ties(self):
        lengths = torch.randint(1, 40, (300,), device=DEVICE)
        gid = torch.repeat_interleave(torch.arange(300, device=DEVICE), lengths)
        vals = torch.randint(0, 3, (len(gid),), device=DEVICE).float()  # many ties

        best = main._segment_argmax(vals, gid, 300)

        expected = [lo + int(vals[lo:hi].argmax()) for lo, hi in segments(lengths)]
        assert best.tolist() == expected


class TestAgentDecide:
    @pytest.fixture
    def stacked(self):
        lengths = torch.randint(1, 60, (200,), device=DEVICE)
        qs = torch.randn(int(lengths.sum()), device=DEVICE)
        return qs, lengths

    def test_greedy_when_eps_zero(self, stacked):
        qs, lengths = stacked
        rows = main.StateNetworkTrainer()._agent_decide(qs, lengths, 0.0)
        expected = [lo + int(qs[lo:hi].argmax()) for lo, hi in segments(lengths)]
        assert rows.tolist() == expected

    def test_random_rows_stay_inside_own_game(self, stacked):
        qs, lengths = stacked
        for _ in range(20):
            rows = main._eps_greedy_rows(qs, lengths, 1.0)
            for row, (lo, hi) in zip(rows.tolist(), segments(lengths)):
                assert lo <= row < hi

    def test_random_rows_cover_whole_game(self):
        lengths = torch.tensor([5, 3], device=DEVICE)
        qs = torch.zeros(8, device=DEVICE)
        seen = {tuple(main._eps_greedy_rows(qs, lengths, 1.0).tolist()) for _ in range(300)}
        assert {r for r, _ in seen} == {0, 1, 2, 3, 4}
        assert {r for _, r in seen} == {5, 6, 7}


class TestForwardData:
    def test_q_is_reward_plus_discounted_value(self):
        vs = bl.VectorizedStates(8)
        _, R, T, B, L, S, _ = vs.states_prim()
        netw = small_net()

        qs = main.StateNetworkTrainer()._forward_data(R, T, B, L, S, netw)

        with torch.no_grad():
            v = netw(B.float(), L.long(), S.float()).view(-1)
        expected = R.float() + (~T).float() * main.DISCOUNT * v
        assert qs.shape == (len(R),)
        assert torch.allclose(qs.float(), expected, atol=1e-4)

    def test_works_when_matmul_dtype_is_half(self, monkeypatch):
        """On CUDA MM_DTYPE is float16 while networks stay float32."""
        vs = bl.VectorizedStates(4)
        _, R, T, B, L, S, _ = vs.states_prim()
        monkeypatch.setattr(bl, "MM_DTYPE", torch.float16)
        main.StateNetworkTrainer()._forward_data(R, T, B, L, S, small_net())


class TestValidation:
    def test_greedy_scores_match_reference(self):
        netw = small_net().eval()
        torch.manual_seed(5)
        got = main._play_greedy_batch(netw, 32)
        torch.manual_seed(5)
        expected = reference_greedy_scores(netw, 32)
        assert torch.equal(got, expected)

    def test_val_net_mean_excludes_penalty_once(self):
        netw = small_net()
        torch.manual_seed(7)
        mean, _ = main.val_net(netw, 32)
        torch.manual_seed(7)
        scores = reference_greedy_scores(netw.eval(), 32) - NEURAL_PENALTY
        assert mean == pytest.approx(scores.float().mean().item())

    def test_greedy_play_with_aux_network(self):
        netw = net.BL_DQN_State_Network_v2(net.BL_DQN_State_Network_Light(), 5).to(DEVICE)
        scores = main._play_greedy_batch(netw.eval(), 8)
        assert scores.shape == (8,)


class TestTrainLoop:
    """Game logic of train_state_network, shared by every state trainer."""

    GAMES = 6

    @pytest.fixture
    def recorded(self, monkeypatch):
        """Run train_state_network on a small net, return its storage."""
        components = small_components()
        trainer = main.StateNetworkTrainer()
        monkeypatch.setattr(trainer, "_get_components", lambda: components)
        monkeypatch.setattr(main, "BATCH_SIZE", 16)
        trainer.train_state_network(games_to_play=3, games_count=self.GAMES)
        return components[2]

    def test_records_chain_into_next_states(self, recorded):
        """Record of game g at loop t+1 starts where loop t ended (or fresh if terminal)."""
        Bs, Ls, Ss, Bf, Lf, Sf, R, T = (b[: recorded.size] for b in recorded.buffers)
        Bs, Bf = Bs.view(-1, BOARD_FLAT), Bf.view(-1, BOARD_FLAT)
        G = self.GAMES
        assert recorded.size % G == 0 and recorded.size > G
        for i in range(recorded.size - G):
            j = i + G  # same game, next loop
            if T[i]:
                assert not Bs[j].any() and not Ss[j].any(), i
            else:
                assert torch.equal(Bs[j], Bf[i]), i
                assert torch.equal(Ls[j], Lf[i]), i
                assert torch.equal(Ss[j], Sf[i]), i

    def test_terminal_records_have_no_legal_moves(self, recorded):
        _, _, _, Bf, Lf, Sf, R, T = (b[: recorded.size] for b in recorded.buffers)
        vs = bl.VectorizedStates(int(T.sum()))
        vs.mass_transition(Bf[T], Lf[T].to(torch.uint8), Sf[T])
        assert not vs.legal_moves().any()
        # Without penalty a terminal move scores < 150 (no empty-board bonus possible).
        assert (R[T] < 150 + NEURAL_PENALTY).all()

    def test_records_are_real_moves(self, recorded):
        """Every afterstate is reachable from its state by one legal move."""
        Bs, Ls, Ss, Bf, Lf, Sf, R, T = (b[:60] for b in recorded.buffers)
        vs = bl.VectorizedStates(60)
        vs.mass_transition(Bs, Ls.to(torch.uint8), Ss)
        _, R2, _, B2, _, S2, MC = vs.states_prim()
        for g, (lo, hi) in enumerate(segments(MC)):
            same = (B2[lo:hi].view(hi - lo, -1) == Bf[g].view(1, -1)).all(1)
            same &= S2[lo:hi, 0] == Sf[g, 0]
            assert same.any(), g


@pytest.mark.parametrize("trainer_cls", STATE_TRAINERS)
class TestStateTrainers:
    def test_network_scores_afterstates(self, trainer_cls):
        trainer = trainer_cls()
        main_net, _, _ = trainer._get_components()
        _, R, T, B, L, S, MC = trainer._gather_data(bl.VectorizedStates(3))
        qs = trainer._forward_data(R, T, B, L, S, main_net)
        rows = trainer._agent_decide(qs, MC, 0.0)
        assert qs.shape == (int(MC.sum()),) and torch.isfinite(qs).all()
        assert rows.shape == (3,)

    def test_gradient_step_updates_main_net_only(self, trainer_cls, monkeypatch):
        monkeypatch.setattr(main, "BATCH_SIZE", 32)
        trainer = trainer_cls()
        main_net, target_net, storage = trainer._get_components()
        storage.add_records(*random_storage_records(4000))
        assert storage.size >= 32
        optimizer = trainer._get_optimizer(main_net, main.LEARING_OPTIM_RATE)
        main_before, target_before = params_snapshot(main_net), params_snapshot(target_net)

        trainer._perform_gradient_descent(main_net, target_net, storage, optimizer)

        main_after = params_snapshot(main_net)
        assert any(not torch.equal(a, b) for a, b in zip(main_before, main_after))
        assert all(torch.isfinite(p).all() for p in main_after)
        assert all(torch.equal(a, b) for a, b in zip(target_before, params_snapshot(target_net)))

    def test_no_gradient_step_on_small_storage(self, trainer_cls):
        trainer = trainer_cls()
        main_net, target_net, storage = trainer._get_components()
        before = params_snapshot(main_net)
        optimizer = trainer._get_optimizer(main_net, main.LEARING_OPTIM_RATE)
        trainer._perform_gradient_descent(main_net, target_net, storage, optimizer)
        assert all(torch.equal(a, b) for a, b in zip(before, params_snapshot(main_net)))


class TestTrainingSideEffects:
    def test_display_info_validates_and_saves(self, monkeypatch, tmp_path, capsys):
        trainer = main.StateNetworkTrainer()
        monkeypatch.setattr(trainer, "_get_components", small_components)
        monkeypatch.setattr(trainer, "SAVE_PATH_NET", tmp_path / "net.pt")
        monkeypatch.setattr(trainer, "SAVE_PATH_SCORE", tmp_path / "score.txt")
        monkeypatch.setattr(main, "GAMES_TO_DISPLAY_INFO", 1)
        monkeypatch.setattr(main, "VALIDATION_GAMES_PER_BATCH", 2)
        monkeypatch.setattr(main, "VALIDATION_GAMES_PER_FIGHT", 2)
        monkeypatch.setattr(main, "BATCH_SIZE", 16)

        trainer.train_state_network(games_to_play=2, games_count=2)

        assert (tmp_path / "net.pt").exists()
        saved = float((tmp_path / "score.txt").read_text())
        assert saved >= 0  # NEURAL_PENALTY already excluded
        assert "Points in val games" in capsys.readouterr().out

    def test_timed_trainer_measures_nested_stages(self, monkeypatch, capsys):
        trainer = main.T_LightTrainer()
        monkeypatch.setattr(trainer, "_get_components", small_components)
        monkeypatch.setattr(main, "BATCH_SIZE", 16)
        originals = (bl.VectorizedStates.states_prim, main.val_net, net.StateStorage.sample)

        with trainer.timed_run("run"):
            trainer.train_state_network(games_to_play=1, games_count=2)

        stats = trainer.profiler.stats
        for stage in ("gather", "forward", "agent_decision", "env_update", "backprop"):
            assert stats[("run", stage)][1] > 0, stage
        # states_prim internals are subsections of the gather stage.
        for part in ("states_prim", "legal_moves", "simulate_transitions", "has_legal_move"):
            path = ("run", "gather", "states_prim") + (() if part == "states_prim" else (part,))
            assert path in stats, part
        assert ("run", "env_update", "storage.add_records") in stats
        assert ("run", "backprop", "storage.sample") in stats
        # Children never take longer than their parent.
        for path, (_, total) in stats.items():
            if len(path) > 1:
                assert total <= stats[path[:-1]][1] + 1e-6, path
        # Everything is restored after the run.
        assert (bl.VectorizedStates.states_prim, main.val_net, net.StateStorage.sample) == originals
        assert "_gather_data" not in vars(trainer)

        trainer.profiler.report()
        out = capsys.readouterr().out
        assert "states_prim" in out and "self %" in out

    def test_profiler_self_times_add_up(self):
        profiler = __import__("profiling").Profiler()
        with profiler.section("a"):
            for _ in range(3):
                with profiler.section("b"):
                    sum(range(20_000))
        (calls_a, total_a), (calls_b, total_b) = profiler.stats[("a",)], profiler.stats[("a", "b")]
        assert calls_a == 1 and calls_b == 3 and total_b <= total_a


class TestShowcase:
    def test_plays_saved_network_until_game_over(self, monkeypatch, tmp_path, capsys):
        trainer = main.StateNetworkTrainer()
        netw = small_net()
        torch.save(netw.state_dict(), tmp_path / "net.pt")
        monkeypatch.setattr(trainer, "_get_components", small_components)
        monkeypatch.setattr(trainer, "SAVE_PATH_NET", tmp_path / "net.pt")
        monkeypatch.setattr(trainer, "SHOWCASE_DELAY", 0)

        trainer.showcase_network()

        out = capsys.readouterr().out
        assert out.count("Score:") > 2
        assert out.rstrip().splitlines()[-1] == "Game Over!"
        final_score = int(out.split("Score: ")[-1].split(",")[0])
        assert final_score >= 0

    def test_reports_missing_weights(self, monkeypatch, tmp_path, capsys):
        trainer = main.StateNetworkTrainer()
        monkeypatch.setattr(trainer, "_get_components", small_components)
        monkeypatch.setattr(trainer, "SAVE_PATH_NET", tmp_path / "missing.pt")
        trainer.showcase_network()
        assert "didn't managed to find network" in capsys.readouterr().out


class TestActionTrainer:
    GAMES = 5

    @pytest.fixture
    def played(self):
        """Storage after 40 eps-greedy moves in GAMES parallel games."""
        storage = net.ActionsStorage(10_000)
        states = bl.VectorizedStates(self.GAMES)
        main_net = net.BL_DQN_Action_Network_v1().to(DEVICE)
        trainer = main.ActionNetworkTrainer()
        for _ in range(40):
            trainer._play_parallel_games(states, main_net, storage, 0.5)
        return storage

    def test_greedy_rows_follow_masked_q_values(self):
        netw = net.BL_DQN_Action_Network_v1().to(DEVICE).eval()
        vs = bl.VectorizedStates(6)
        A, _, _, B, L, S, MC = vs.states_prim()
        rows = main._eps_greedy_rows(main._move_values(netw, vs, A, B, L, S, MC), MC, 0.0)
        with torch.no_grad():
            qs = netw(vs.boards.view(-1, 1, BOARD_LEN, BOARD_LEN).float(), vs.available.long(), vs.streaks.float())
        qs[~vs.legal_moves()] = -torch.inf
        assert torch.equal(A[rows], qs.argmax(dim=1))

    def test_val_net_and_timed_training(self, monkeypatch, tmp_path):
        monkeypatch.setattr(main, "BATCH_SIZE", 16)
        monkeypatch.setattr(main.ActionNetworkTrainer, "SAVE_PATH", tmp_path / "a.pt")
        mean, std = main.val_net(net.BL_DQN_Action_Network_v1().to(DEVICE), 4)
        assert mean >= 0 and std >= 0
        trainer = main.T_ActionNetworkTrainer()
        with trainer.timed_run("run"):
            trainer.train_action_network(games_to_play=1, games_count=4)
        assert (tmp_path / "a.pt").exists()
        assert {("run", "play"), ("run", "play", "states_prim"), ("run", "backprop")} <= set(trainer.profiler.stats)

    def test_records_chain_and_masks_match(self, played):
        Bs, Ls, Ss, Bf, Lf, Sf, A, R, T, Mf = (b[: played.size] for b in played.buffers)
        G = self.GAMES
        assert played.size == 40 * G
        for i in range(played.size - G):
            j = i + G
            if T[i]:
                assert not Bs[j].any() and not Ss[j].any(), i
            else:
                assert torch.equal(Bs[j], Bf[i]) and torch.equal(Ls[j], Lf[i]), i
        vs = bl.VectorizedStates(played.size)
        vs.mass_transition(Bf, Lf.to(torch.uint8), Sf)
        assert torch.equal(vs.legal_moves(), Mf)
        assert not Mf[T].any() and Mf[~T].any(dim=1).all()

    def test_actions_produce_stored_afterstates(self, played):
        Bs, Ls, Ss, Bf, _, Sf, A, R, T, _ = (b[:40] for b in played.buffers)
        vs = bl.VectorizedStates(40)
        vs.mass_transition(Bs, Ls.to(torch.uint8), Ss)
        A2, _, _, B2, _, S2, MC = vs.states_prim()
        for g, (lo, hi) in enumerate(segments(MC)):
            row = lo + int((A2[lo:hi] == A[g].long()).nonzero())
            assert torch.equal(B2[row], Bf[g]) and S2[row, 0] == Sf[g, 0], g

    def test_storage_wraps_around(self):
        storage = net.ActionsStorage(10)
        for start in (0, 7):
            n = 7
            storage.add_records(
                torch.zeros(n, BOARD_FLAT, dtype=torch.bool),
                torch.zeros(n, BLOCKS_TO_PICK, dtype=torch.uint8),
                torch.zeros(n, 1, dtype=torch.bool),
                torch.zeros(n, 1, BOARD_LEN, BOARD_LEN, dtype=torch.bool),
                torch.zeros(n, BLOCKS_TO_PICK, dtype=torch.uint8),
                torch.zeros(n, 1, dtype=torch.bool),
                torch.arange(start, start + n),
                torch.arange(start, start + n),
                torch.zeros(n, dtype=torch.bool),
                torch.zeros(n, bl.NN_OUTPUT_FLAT, dtype=torch.bool),
            )
        assert storage.size == 10 and storage.offset == 4
        assert storage.buffers[7].tolist() == [10, 11, 12, 13, 4, 5, 6, 7, 8, 9]
