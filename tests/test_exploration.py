"""Exploration drawn before the forward pass: only moves of exploiting games
are valued by the network, _agent_decide merges both groups back."""

import pytest
import torch

import blokudoku_lib as bl
import main
from blokudoku_lib import DEVICE
from test_training import segments, small_components


@pytest.fixture
def trainer():
    return main.StateNetworkTrainer()


@pytest.fixture
def stacked():
    """(qs, lengths) of 200 games with 1..59 moves each."""
    lengths = torch.randint(1, 60, (200,), device=DEVICE)
    qs = torch.randn(int(lengths.sum()), device=DEVICE)
    return qs, lengths


def plain_argmax(qs, lengths):
    return [lo + int(qs[lo:hi].argmax()) for lo, hi in segments(lengths)]


class TestDrawExploration:
    def test_eps_zero_keeps_every_row(self, trainer, stacked):
        qs, lengths = stacked
        explore, keep = trainer._draw_exploration(lengths, 0.0)
        assert explore.shape == (200,) and explore.dtype == torch.bool
        assert not explore.any()
        assert torch.equal(keep, torch.arange(len(qs), device=DEVICE))

    def test_eps_one_keeps_no_row(self, trainer, stacked):
        _, lengths = stacked
        explore, keep = trainer._draw_exploration(lengths, 1.0)
        assert explore.all()
        assert keep.shape == (0,) and keep.dtype == torch.long

    def test_kept_rows_are_whole_segments_of_exploiting_games(self, trainer, stacked):
        _, lengths = stacked
        explore, keep = trainer._draw_exploration(lengths, 0.5)
        assert 20 < int(explore.sum()) < 180
        expected = [
            row
            for g, (lo, hi) in enumerate(segments(lengths))
            if not explore[g]
            for row in range(lo, hi)
        ]
        assert keep.tolist() == expected

    def test_exploration_rate_matches_eps(self, trainer):
        lengths = torch.ones(20_000, dtype=torch.long, device=DEVICE)
        explore, _ = trainer._draw_exploration(lengths, 0.3)
        assert abs(explore.float().mean().item() - 0.3) < 0.02


class TestAgentDecide:
    def test_all_exploit_is_plain_argmax(self, trainer, stacked):
        qs, lengths = stacked
        explore, keep = trainer._draw_exploration(lengths, 0.0)
        rows = trainer._agent_decide(qs, keep, lengths, explore)
        assert rows.dtype == torch.long
        assert rows.tolist() == plain_argmax(qs, lengths)

    def test_first_of_tied_maxima(self, trainer):
        lengths = torch.tensor([4, 3], device=DEVICE)
        qs = torch.tensor([1.0, 2.0, 2.0, 0.0, 5.0, 5.0, 5.0], device=DEVICE)
        explore, keep = trainer._draw_exploration(lengths, 0.0)
        assert trainer._agent_decide(qs, keep, lengths, explore).tolist() == [1, 4]

    def test_all_explore_needs_no_values(self, trainer, stacked):
        _, lengths = stacked
        explore, keep = trainer._draw_exploration(lengths, 1.0)
        for _ in range(20):
            rows = trainer._agent_decide(torch.empty(0, device=DEVICE), keep, lengths, explore)
            for row, (lo, hi) in zip(rows.tolist(), segments(lengths)):
                assert lo <= row < hi

    def test_exploring_rows_cover_whole_game(self, trainer):
        lengths = torch.tensor([5, 3], device=DEVICE)
        explore, keep = trainer._draw_exploration(lengths, 1.0)
        qs = torch.empty(0, device=DEVICE)
        seen = {
            tuple(trainer._agent_decide(qs, keep, lengths, explore).tolist())
            for _ in range(300)
        }
        assert {r for r, _ in seen} == {0, 1, 2, 3, 4}
        assert {r for _, r in seen} == {5, 6, 7}

    def test_mixed_games_merge_both_rules(self, trainer, stacked):
        qs, lengths = stacked
        explore, keep = trainer._draw_exploration(lengths, 0.5)
        best = plain_argmax(qs, lengths)

        rows = trainer._agent_decide(qs[keep], keep, lengths, explore)

        for g, (row, (lo, hi)) in enumerate(zip(rows.tolist(), segments(lengths))):
            assert lo <= row < hi, g
            if not explore[g]:
                assert row == best[g], g

    def test_values_of_exploring_games_never_matter(self, trainer, stacked):
        """Garbage values at exploring rows must not change exploiting games."""
        qs, lengths = stacked
        explore, keep = trainer._draw_exploration(lengths, 0.5)
        torch.manual_seed(1)
        rows_a = trainer._agent_decide(qs[keep], keep, lengths, explore)
        torch.manual_seed(1)
        poisoned = qs.clone()
        poisoned[torch.repeat_interleave(explore, lengths)] = 1e9
        rows_b = trainer._agent_decide(poisoned[keep], keep, lengths, explore)
        assert torch.equal(rows_a, rows_b)


class TestTrainingLoopForwardsOnlyExploitingMoves:
    @pytest.fixture
    def counting_trainer(self, monkeypatch):
        """Light trainer with recorded (rows forwarded, rows gathered) per loop."""
        trainer = main.LightTrainer()
        monkeypatch.setattr(trainer, "_get_components", small_components)
        monkeypatch.setattr(main, "BATCH_SIZE", 16)
        monkeypatch.setattr(main, "GAMES_TO_DISPLAY_INFO", 10**9)
        forwarded, gathered = [], []
        gather, forward = trainer._gather_data, trainer._forward_data

        def counting_gather(states):
            out = gather(states)
            gathered.append(int(out[-1].sum()))
            return out

        def counting_forward(R, *args):
            forwarded.append(len(R))
            return forward(R, *args)

        monkeypatch.setattr(trainer, "_gather_data", counting_gather)
        monkeypatch.setattr(trainer, "_forward_data", counting_forward)
        return trainer, forwarded, gathered

    def test_pure_exploration_skips_network_entirely(self, counting_trainer, monkeypatch):
        trainer, forwarded, gathered = counting_trainer
        monkeypatch.setattr(main, "EPS_MAX", 1.0)
        monkeypatch.setattr(main, "EPS_MIN", 1.0)
        trainer.train_state_network(games_to_play=1, games_count=4)
        assert len(forwarded) == len(gathered) > 3
        assert forwarded == [0] * len(forwarded)

    def test_pure_exploitation_values_every_move(self, counting_trainer, monkeypatch):
        trainer, forwarded, gathered = counting_trainer
        monkeypatch.setattr(main, "EPS_MAX", 0.0)
        monkeypatch.setattr(main, "EPS_MIN", 0.0)
        trainer.train_state_network(games_to_play=1, games_count=4)
        assert len(forwarded) == len(gathered) > 3
        assert forwarded == gathered

    def test_partial_exploration_forwards_a_subset(self, counting_trainer, monkeypatch):
        trainer, forwarded, gathered = counting_trainer
        monkeypatch.setattr(main, "EPS_MAX", 0.5)
        monkeypatch.setattr(main, "EPS_MIN", 0.5)
        trainer.train_state_network(games_to_play=2, games_count=8)
        assert all(0 <= f <= g for f, g in zip(forwarded, gathered))
        assert 0.2 < sum(forwarded) / sum(gathered) < 0.8
