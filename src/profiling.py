"""Nested wall-clock profiling of chosen functions, used by `time_profile`."""

import functools
import time
from contextlib import contextmanager
import torch
import blokudoku_lib as bl

class Profiler:
    """Times named sections, which may be nested.

    Sections are stored under their full path (outer first), so the report is
    a tree: time of a section includes its children, "self" is what is left.
    On CUDA every boundary synchronizes, so time lands in the section that
    launched the kernels (the profiled run gets slightly slower).
    """

    def __init__(self):
        self.stats = {}  # path tuple -> [calls, total seconds]
        self._stack = []

    @contextmanager
    def section(self, name: str):
        """Time the `with` block as section `name`."""
        _sync()
        self._stack.append(name)
        path, start = tuple(self._stack), time.perf_counter()
        try:
            yield
        finally:
            _sync()
            self._stack.pop()
            entry = self.stats.setdefault(path, [0, 0.0])
            entry[0] += 1
            entry[1] += time.perf_counter() - start

    @contextmanager
    def instrument(self, targets: list):
        """Temporarily replace functions with timed versions.

        Args:
            targets: [(owner, {attribute: section name})], owner being a class,
            instance or module. Attributes owner lacks are skipped.
        """
        saved = []
        try:
            for owner, names in targets:
                for attr, section in names.items():
                    if hasattr(owner, attr):
                        saved.append((owner, attr, vars(owner).get(attr)))
                        setattr(owner, attr, self._timed(getattr(owner, attr), section))
            yield self
        finally:
            for owner, attr, raw in saved:
                setattr(owner, attr, raw) if raw else delattr(owner, attr)

    def _timed(self, fn, name: str):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with self.section(name):
                return fn(*args, **kwargs)

        return wrapper

    def report(self):
        """Print the tree of sections, siblings sorted by time spent."""
        run = sum(t for path, (_, t) in self.stats.items() if len(path) == 1)
        if not run:
            print("Nothing was measured.")
            return
        print(f"{'section':<46}{'calls':>8}{'total s':>10}{'% run':>8}{'self %':>8}{'avg ms':>9}")
        for path in self._tree_order(()):
            calls, total = self.stats[path]
            own = total - sum(t for p, (_, t) in self.stats.items() if p[:-1] == path)
            label = "  " * (len(path) - 1) + path[-1]
            print(f"{label:<46}{calls:>8}{total:>10.3f}{100 * total / run:>8.2f}"
                  f"{100 * own / run:>8.2f}{1000 * total / calls:>9.3f}")  # fmt: skip

    def _tree_order(self, prefix: tuple):
        kids = sorted((p for p in self.stats if p[:-1] == prefix), key=lambda p: -self.stats[p][1])
        return [q for p in kids for q in (p, *self._tree_order(p))]


def _sync():
    if bl.DEVICE.type == "cuda":
        torch.cuda.synchronize()
