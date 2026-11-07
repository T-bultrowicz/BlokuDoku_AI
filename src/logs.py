"""Terminal logging of training, validation and testing progress.

Every line is `[tag]` padded to LOG_TAG_WIDTH, then fields joined by ' | '.
"""

import torch

LOG_TAG_WIDTH = 10

def log(tag: str, *fields: str):
    """Print one log line: fixed-width [tag] followed by fields joined by ' | '."""
    print(f"[{tag}]".ljust(LOG_TAG_WIDTH) + " | ".join(fields))


def td_error_stats(predicted: torch.Tensor, actual: torch.Tensor):
    """Mean, 10th and 90th percentile of |predicted - actual|.

    Args:
        predicted: (B,) values the network predicts.
        actual: (B,) target values of the same samples.

    Returns:
        Tuple (mean, p10, p90) of python floats.
    """
    errors = (predicted.detach() - actual.detach()).abs().float()
    p10, p90 = torch.quantile(errors, torch.tensor([0.1, 0.9], device=errors.device))
    return errors.mean().item(), p10.item(), p90.item()


def log_progress(games: int, games_in_batch: int, elapsed: float, eps: float):
    """Log training pace: games played so far, games_in_batch played in elapsed seconds."""
    games_per_second = games_in_batch / elapsed if elapsed else float("inf")
    log("train", f"games {games:7d}", f"{games_per_second:6.2f} games/s",
        f"batch {elapsed:6.2f}s", f"eps {eps:.4f}")  # fmt: skip


def log_validation(games: int, mean: float, std: float):
    """Log result of validation games played during training."""
    log("val", f"games {games:7d}", f"mean {mean:9.2f}", f"std {std:9.2f}")


def log_td_error(predicted: torch.Tensor, actual: torch.Tensor):
    """Log |predicted - actual| stats of a batch under the `td err` tag."""
    mean, p10, p90 = td_error_stats(predicted, actual)
    log("td err", f"mean {mean:9.2f}", f"p10 {p10:9.2f}", f"p90 {p90:9.2f}")
