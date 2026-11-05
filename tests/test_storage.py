import numpy as np
import pytest

import blokudoku_lib as bl
import network_lib as net


def records(n: int, fill: float = 0.3, rewards: np.ndarray | None = None):
    """Random batch of n records in the add_records() argument order."""

    def boards():
        return np.random.rand(n, 1, bl.BOARD_LEN, bl.BOARD_LEN) < fill

    def blocks():
        return np.random.randint(0, bl.BLOCKS_SIZE, (n, bl.BLOCKS_TO_PICK)).astype(
            np.int8
        )

    def streaks():
        return np.random.rand(n, 1) < 0.5

    if rewards is None:
        rewards = np.arange(n, dtype=np.int16)
    return (boards(), blocks(), streaks(), boards(), blocks(), streaks(),
            rewards, np.random.rand(n) < 0.5)  # fmt: skip


REWARDS = 6  # index of rewards in the records tuple


class TestStateStorage:
    def test_rejects_empty_capacity(self):
        with pytest.raises(ValueError):
            net.StateStorage(0)

    def test_wraps_around_keeping_order(self):
        s = net.StateStorage(10)
        first = records(7)
        s.add_records(*first)
        second = records(6, rewards=np.arange(100, 106, dtype=np.int16))
        s.add_records(*(bl.to_tensor(x) for x in second))  # tensor input, float dtype

        expected = [103, 104, 105, 3, 4, 5, 6, 100, 101, 102]
        assert s.buffers[REWARDS].tolist() == expected
        assert s.offset == 3 and s.size == 10
        assert np.array_equal(s.buffers[0][7:].cpu().numpy(), second[0][:3])

    def test_accepts_flat_boards(self):
        s = net.StateStorage(10)
        recs = records(7)
        s.add_records(recs[0].reshape(7, -1), *recs[1:])
        assert np.array_equal(s.buffers[0][:7].cpu().numpy(), recs[0])

    def test_batch_larger_than_capacity_keeps_newest(self):
        s = net.StateStorage(4)
        s.add_records(*records(11))
        assert sorted(s.buffers[REWARDS].tolist()) == [7, 8, 9, 10]
        assert s.offset == 3 and s.size == 4

    def test_empty_batch_is_noop(self):
        s = net.StateStorage(4)
        s.add_records(*records(0))
        assert s.offset == 0 and s.size == 0

    def test_sample_shapes(self):
        s = net.StateStorage(100)
        s.add_records(*records(50))
        out = s.sample(16)
        assert len(out) == len(s.buffers)
        for t, buf in zip(out, s.buffers):
            assert t.shape == (16, *buf.shape[1:]) and t.dtype == buf.dtype
            assert t.device.type == bl.DEVICE.type


class TestSmartStateStorage:
    BUCKETS = (
        ("empty_boards", -1, net.SmartStateStorage.EMPTY_BOARDS_MAX),
        ("casual_boards", net.SmartStateStorage.EMPTY_BOARDS_MAX, net.SmartStateStorage.CASUAL_BOARDS_MAX),
        ("crowded_boards", net.SmartStateStorage.CASUAL_BOARDS_MAX, net.SmartStateStorage.CROWDED_BOARDS_MAX),
        ("ultra_crowded_boards", net.SmartStateStorage.CROWDED_BOARDS_MAX, bl.BOARD_FLAT),
    )  # fmt: skip

    @pytest.fixture
    def filled(self):
        n = 4000
        recs = records(n)
        # Vary fill per board, so every bucket gets records.
        boards = np.random.rand(n, 1, bl.BOARD_LEN, bl.BOARD_LEN) < np.random.rand(
            n, 1, 1, 1
        )
        recs = (boards,) + recs[1:]
        storage = net.SmartStateStorage()
        storage.add_records(*recs)
        return storage, boards.reshape(n, -1).sum(1)

    def test_records_land_in_right_bucket(self, filled):
        storage, counts = filled
        for name, lo, hi in self.BUCKETS:
            bucket = getattr(storage, name)
            stored = bucket.buffers[0][: bucket.size].reshape(bucket.size, -1).sum(1)
            assert ((stored > lo) & (stored <= hi)).all(), name
            assert bucket.size == ((counts > lo) & (counts <= hi)).sum(), name

    def test_sample_is_stratified(self, filled):
        storage, _ = filled
        out = storage.sample(64)
        for t in out:
            assert t.shape[0] == 64 and t.device.type == bl.DEVICE.type
