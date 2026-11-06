import numpy as np
import pytest
import torch

import blokudoku_lib as bl
import network_lib as net
import reference as ref
from blokudoku_lib import (
    BLOCKS_SIZE,
    BLOCKS_TO_PICK,
    BOARD_FLAT,
    BOARD_LEN,
    DEVICE,
    EMPTY_SLOT,
    NEURAL_PENALTY,
    NN_OUTPUT_FLAT,
    VectorizedStates,
)
from conftest import to_numpy


class TestConstantTables:
    def test_single_cell_block_placement(self):
        expected = np.zeros(BOARD_FLAT, bool)
        expected[0] = True
        assert np.array_equal(VectorizedStates.PLACEMENTS[0, 0].cpu().numpy(), expected)

    def test_single_cell_block_fits_everywhere(self):
        assert VectorizedStates.IN_BOUNDS[0].all()

    def test_empty_slot_fits_nowhere(self):
        assert not VectorizedStates.IN_BOUNDS[EMPTY_SLOT].any()

    def test_lines_are_rows_columns_and_squares(self):
        lines = VectorizedStates.LINES.cpu().numpy().reshape(-1, BOARD_LEN, BOARD_LEN)
        assert (lines.sum(axis=(1, 2)) == BOARD_LEN).all()
        assert lines[0][:, 0].all()  # column 0
        assert lines[BOARD_LEN][0, :].all()  # row 0
        assert lines[2 * BOARD_LEN][:3, :3].all()  # top-left square


class TestInit:
    @pytest.mark.parametrize("n", [0, -1, VectorizedStates.MAX_BOARDS + 1])
    def test_rejects_bad_sizes(self, n):
        with pytest.raises(ValueError):
            VectorizedStates(n)

    def test_fresh_games(self):
        vs = VectorizedStates(7)
        assert vs.boards.shape == (7, BOARD_FLAT) and not vs.boards.any()
        assert (
            vs.available.shape == (7, BLOCKS_TO_PICK)
            and vs.available.dtype == torch.uint8
        )
        assert (vs.available < BLOCKS_SIZE).all()
        assert vs.streaks.shape == (7, 1) and not vs.streaks.any()


class TestLegalMoves:
    @pytest.mark.parametrize("fill", [0.0, 0.2, 0.5, 0.9])
    @pytest.mark.parametrize("method", ["legal_moves", "legal_moves2"])
    def test_matches_reference(self, random_states, fill, method):
        vs = random_states(500, fill)
        boards, available = to_numpy(vs.boards, vs.available)
        expected = np.array(
            [
                ref.valid_mask(b.reshape(BOARD_LEN, BOARD_LEN), a)
                for b, a in zip(boards, available)
            ]
        )

        out = getattr(vs, method)()

        assert out.shape == (vs.N, NN_OUTPUT_FLAT)
        assert out.dtype == torch.bool
        assert np.array_equal(out.cpu().numpy(), expected)


class TestStatesPrim:
    N = 300

    @pytest.fixture
    def prim(self, random_states):
        """Random valid states, a third with one block left, with their states_prim()."""
        vs = random_states(self.N, 0.5, valid=True)
        avail = vs.available.cpu().numpy()
        for g in np.flatnonzero(np.random.rand(self.N) < 1 / 3):
            avail[g] = EMPTY_SLOT
            avail[g, np.random.randint(BLOCKS_TO_PICK)] = np.random.randint(BLOCKS_SIZE)
        vs.available = bl.to_tensor(avail, torch.uint8)
        return vs, vs.states_prim()

    @staticmethod
    def segments(MC):
        """(start, stop) row range of every game inside the stacked tensors."""
        stops = np.cumsum(MC)
        return list(zip(stops - MC, stops))

    def test_shapes_and_dtypes(self, prim):
        _, (A, R, T, B, L, S, MC) = prim
        m = int(MC.sum())
        assert MC.shape == (self.N,) and MC.dtype == torch.long
        assert A.shape == R.shape == T.shape == (m,)
        assert B.shape == (m, 1, BOARD_LEN, BOARD_LEN)
        assert L.shape == (m, BLOCKS_TO_PICK)
        assert S.shape == (m, 1)
        assert A.dtype == torch.long and R.dtype == torch.int16
        assert T.dtype == S.dtype == B.dtype == torch.bool
        assert L.dtype == torch.uint8
        for t in (A, R, T, B, L, S, MC):
            assert t.device.type == DEVICE.type

    def test_move_counts_and_game_major_order(self, prim):
        vs, (A, _, _, _, _, _, MC) = prim
        assert torch.equal(MC, vs.legal_moves().sum(dim=-1))
        assert (MC == 0).sum() > 0  # some games are stuck
        A, MC = to_numpy(A, MC)
        for g, (lo, hi) in enumerate(self.segments(MC)):
            expected = np.flatnonzero(vs.legal_moves()[g].cpu().numpy())
            assert np.array_equal(A[lo:hi], expected), g

    def test_nn_and_storage_ready(self, prim):
        _, (_, _, _, B, L, S, _) = prim
        boards, streaks = B.to(bl.MM_DTYPE), S.to(bl.MM_DTYPE)
        assert boards.dtype == streaks.dtype == bl.MM_DTYPE
        assert torch.equal(boards.bool(), B) and torch.equal(streaks.bool(), S)
        storage = net.StateStorage(len(B))
        storage.add_records(B, L, S, B, L, S, torch.zeros(len(B)), S.view(-1))
        assert storage.size == len(B)

    def test_matches_reference(self, prim):
        vs, out = prim
        boards, avail, streaks = to_numpy(vs.boards, vs.available, vs.streaks)
        boards = boards.reshape(self.N, BOARD_LEN, BOARD_LEN)
        A, R, T, B, L, S, MC = to_numpy(*out)
        B = B.reshape(len(B), BOARD_FLAT)

        checked = 0
        for g, (lo, hi) in enumerate(self.segments(MC)):
            assert np.array_equal(np.flatnonzero(ref.valid_mask(boards[g], avail[g])), A[lo:hi]), g
            for i in range(lo, hi):
                a = int(A[i])
                r, board, blocks, streak = ref.transition(
                    boards[g], avail[g], streaks[g, 0], a
                )
                assert np.array_equal(board.reshape(-1), B[i]), (g, a)
                assert streak == S[i, 0], (g, a)
                if (blocks == EMPTY_SLOT).all():
                    assert (L[i] < BLOCKS_SIZE).all(), (g, a)
                else:
                    assert np.array_equal(blocks, L[i]), (g, a)
                # Terminal and its penalty depend on drawn blocks, so use L.
                assert T[i] == (not ref.valid_mask(board, L[i]).any()), (g, a)
                assert R[i] == r + (NEURAL_PENALTY if T[i] else 0), (g, a)
                checked += 1
        assert checked > 1000

    def test_empty_slots_moved_last(self, prim):
        _, (_, _, _, _, L, _, _) = prim
        empty = L == EMPTY_SLOT
        assert (empty[:, :-1] <= empty[:, 1:]).all()

    def test_one_draw_of_new_blocks_per_game(self, prim):
        vs, (_, _, _, _, L, _, MC) = prim
        one_left = (vs.available == EMPTY_SLOT).sum(1) == BLOCKS_TO_PICK - 1
        assert one_left.sum() > 50
        MC = MC.cpu().numpy()
        for g, (lo, hi) in enumerate(self.segments(MC)):
            if one_left[g] and hi > lo:  # the last block may not fit anywhere
                assert (L[lo:hi] == L[lo]).all(), g


class TestTransitions:
    @pytest.fixture
    def stepped(self, random_states):
        """States after playing one random legal action per playable game."""
        vs = random_states(64, 0.4, valid=True)
        _, _, T, B, L, S, MC = vs.states_prim()
        playable = MC > 0
        # Random row inside every playable game's segment.
        offsets = MC.cumsum(0) - MC
        rows = offsets + (torch.rand(vs.N, device=DEVICE) * MC).long()
        rows = rows[playable]

        boards, avail, streaks = vs.boards.clone(), vs.available.clone(), vs.streaks.clone()
        boards[playable] = B[rows].view(-1, BOARD_FLAT)
        avail[playable] = L[rows]
        streaks[playable] = S[rows]
        vs.mass_transition(boards.view(-1, 1, BOARD_LEN, BOARD_LEN), avail, streaks.view(-1))

        terminal = torch.zeros(vs.N, dtype=torch.bool, device=DEVICE)
        terminal[playable] = T[rows]
        return vs, playable, terminal

    def test_mass_transition_keeps_shapes(self, stepped):
        vs, _, _ = stepped
        assert vs.boards.shape == (64, BOARD_FLAT)
        assert vs.available.shape == (64, BLOCKS_TO_PICK)
        assert vs.streaks.shape == (64, 1)

    def test_terminal_agrees_with_new_state(self, stepped):
        vs, playable, terminal = stepped
        assert playable.sum() > 32  # unplayable games got a random illegal action
        assert torch.equal(vs.legal_moves().any(1)[playable], ~terminal[playable])

    def test_reset_only_masked_games(self, stepped):
        vs, _, _ = stepped
        before = (vs.boards.clone(), vs.available.clone(), vs.streaks.clone())
        done = torch.rand(64, device=DEVICE) < 0.5

        vs.reset(done)

        assert not vs.boards[done].any() and not vs.streaks[done].any()
        assert (vs.available[done] < BLOCKS_SIZE).all()
        assert vs.available.dtype == torch.uint8
        for now, prev in zip((vs.boards, vs.available, vs.streaks), before):
            assert torch.equal(now[~done], prev[~done])

    def test_drop_removes_games(self, stepped):
        vs, _, _ = stepped
        before = (vs.boards.clone(), vs.available.clone(), vs.streaks.clone())
        dropped = torch.rand(64, device=DEVICE) < 0.5

        vs.drop(dropped)

        assert vs.N == int((~dropped).sum())
        for now, prev in zip((vs.boards, vs.available, vs.streaks), before):
            assert torch.equal(now, prev[~dropped])
        _, _, _, _, _, _, MC = vs.states_prim()
        assert MC.shape == (vs.N,)

    def test_drop_all_is_empty_batch(self, stepped):
        vs, _, _ = stepped
        vs.drop(torch.ones(64, dtype=torch.bool, device=DEVICE))
        assert vs.N == 0
        assert vs.legal_moves().shape == (0, NN_OUTPUT_FLAT)
