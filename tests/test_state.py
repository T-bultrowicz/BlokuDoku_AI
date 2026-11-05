import numpy as np
import pytest

import reference as ref
from blokudoku_lib import (
    BLOCKS_TO_PICK,
    BOARD_LEN,
    NEURAL_PENALTY,
    NN_OUTPUT_FLAT,
    State,
)


class TestActionCoding:
    @pytest.mark.parametrize("action", [0, 80, 81, 242])
    def test_roundtrip(self, action):
        assert State.encode_action(*State.decode_action(action)) == action

    def test_decode(self):
        assert State.decode_action(81 + 2 * 9 + 5) == (1, 2, 5)


class TestState:
    def test_fresh_game_io_shapes(self):
        st = State()
        assert st.in_board().shape == (1, BOARD_LEN, BOARD_LEN)
        assert st.in_board().dtype == np.bool_ and not st.in_board().any()
        assert (
            st.in_blocks().shape == (BLOCKS_TO_PICK,)
            and st.in_blocks().dtype == np.int8
        )
        assert st.in_streak().shape == (1,) and st.in_streak().dtype == np.bool_
        assert st.calculate_mask().shape == (NN_OUTPUT_FLAT,)

    def test_all_1move_states_shapes(self):
        st = State()
        mask = st.calculate_mask()
        acts, rews, terms, boards, blocks, streaks = st.all_1move_states(mask)
        n = len(acts)
        assert np.array_equal(acts, np.flatnonzero(mask))
        assert rews.shape == terms.shape == (n,)
        assert boards.shape == (n, 1, BOARD_LEN, BOARD_LEN)
        assert blocks.shape == (n, BLOCKS_TO_PICK) and blocks.dtype == np.int8
        assert streaks.shape == (n, 1)

    def test_illegal_move_raises(self):
        st = State()
        st._vs.boards[0, -1] = True  # every block placed at (8, 8) covers that cell
        with pytest.raises(ValueError):
            st.transition(0, BOARD_LEN - 1, BOARD_LEN - 1)

    @pytest.mark.parametrize("game", range(10))
    def test_random_game_matches_reference(self, game):
        st = State()
        mask = st.calculate_mask()
        game_over = False
        while not game_over:
            board, blocks, streak = (
                st.in_board()[0],
                st.in_blocks(),
                bool(st.in_streak()[0]),
            )
            assert np.array_equal(mask, ref.valid_mask(board, blocks))

            a = int(np.random.choice(np.flatnonzero(mask)))
            r_ref, board_ref, _, streak_ref = ref.transition(board, blocks, streak, a)
            reward, game_over, mask = st.transition(*State.decode_action(a))

            assert np.array_equal(st.in_board()[0], board_ref)
            assert st.in_streak()[0] == streak_ref
            assert game_over == (not mask.any())
            assert reward == r_ref + (NEURAL_PENALTY if game_over else 0)
