# Commenting rules
- Docstrings in Google style: one-line summary, longer description mostly skippable, then Args/Returns sections.
- Comment mostly just what is needed to understand function as black box, as in functional paradigm.
- Docstring informs about shapes functions return, and take as args if it is not very clear.
- Commends inside functions should inform about non-obvious shapes. Prefer numbers than aliases in comments.
- Use `# fmt: off` / `# fmt: on` around hand-aligned signatures.
- Example comment for complicated function:
```python
    def _simulate_transitions(
        self, boards: torch.Tensor, placed: torch.Tensor, old_streaks: torch.Tensor
    ):
        """Calculate all logic of making legal moves: line clearings, rewards
        and streaks for M arbitrary afterstates.

        Args:
            boards: (M, BOARD_FLAT) bools, boards right after placing a block.
            placed: (M, BOARD_FLAT) bools, cells covered by the placed block.
            old_streaks: (M,) bools, streak flags before the move.

        Returns:
            Tuple (rewards, boards, streaks) of shapes (M,), (M, BOARD_FLAT), (M,):
            reward without NEURAL_PENALTY, board after clearings, new streak flag.
        """
```
- Example comment for simple function:
```python
    def legal_moves(self):
        """Calculate mask of legal moves.

        Returns:
            mask: of (N, BLOCKS_TO_PICK * BOARD_FLAT) bools, where mask[gid, id]
            informs whether move encoded as id is legal to play on game #gid.
        """
```