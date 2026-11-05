import numpy as np
import pytest
import torch

import blokudoku_lib as bl
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

    def test_shapes_and_dtypes(self, prim):
        _, (A, R, T, B, L, S) = prim
        n = self.N
        assert A.shape == T.shape == S.shape == R.shape == (n, NN_OUTPUT_FLAT)
        assert B.shape == (n, NN_OUTPUT_FLAT, BOARD_FLAT)
        assert L.shape == (n, NN_OUTPUT_FLAT, BLOCKS_TO_PICK)
        assert A.dtype == T.dtype == S.dtype == B.dtype == torch.bool
        assert R.dtype == torch.int16 and L.dtype == torch.uint8

    def test_illegal_actions_are_zeroed(self, prim):
        _, (A, R, T, _, _, S) = prim
        assert not (R[~A].any() or T[~A].any() or S[~A].any())

    def test_matches_reference(self, prim):
        vs, out = prim
        boards, avail, streaks = to_numpy(vs.boards, vs.available, vs.streaks)
        boards = boards.reshape(self.N, BOARD_LEN, BOARD_LEN)
        A, R, T, B, L, S = to_numpy(*out)

        checked = 0
        for g in range(self.N):
            assert np.array_equal(ref.valid_mask(boards[g], avail[g]), A[g]), g
            for a in np.flatnonzero(A[g]):
                r, board, blocks, streak = ref.transition(
                    boards[g], avail[g], streaks[g, 0], int(a)
                )
                assert np.array_equal(board.reshape(-1), B[g, a]), (g, a)
                assert streak == S[g, a], (g, a)
                if (blocks == EMPTY_SLOT).all():
                    assert (L[g, a] < BLOCKS_SIZE).all(), (g, a)
                else:
                    assert np.array_equal(blocks, L[g, a]), (g, a)
                # Terminal and its penalty depend on drawn blocks, so use L.
                assert T[g, a] == (not ref.valid_mask(board, L[g, a]).any()), (g, a)
                assert R[g, a] == r + (NEURAL_PENALTY if T[g, a] else 0), (g, a)
                checked += 1
        assert checked > 1000

    def test_one_draw_of_new_blocks_per_game(self, prim):
        vs, (A, _, _, _, L, _) = prim
        one_left = (vs.available == EMPTY_SLOT).sum(1) == BLOCKS_TO_PICK - 1
        assert one_left.sum() > 50
        for g in one_left.nonzero().flatten():
            draws = L[g][A[g]]
            if len(draws):  # the last block may not fit anywhere
                assert (draws == draws[0]).all(), g


class TestTransitions:
    @pytest.fixture
    def stepped(self, random_states):
        """States after playing one random legal action per game."""
        vs = random_states(64, 0.4, valid=True)
        A, _, T, B, L, S = vs.states_prim()
        g = torch.arange(vs.N, device=DEVICE)
        actions = torch.multinomial(A.float() + 1e-9, 1).squeeze(1)
        vs.mass_transition(B[g, actions], L[g, actions], S[g, actions].view(-1, 1))
        return vs, A.any(1), T[g, actions]

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

    def test_nn_input(self, stepped):
        vs, _, _ = stepped
        boards, blocks, streaks = vs.nn_input()
        assert (
            boards.shape == (64, 1, BOARD_LEN, BOARD_LEN) and boards.dtype == torch.bool
        )
        assert blocks.shape == (64, BLOCKS_TO_PICK)
        assert streaks.shape == (64, 1) and streaks.dtype == torch.bool
        assert torch.equal(boards.reshape(64, -1), vs.boards)
