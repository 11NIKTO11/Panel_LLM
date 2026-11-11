"""
How much a statistic of a group of subjects would move with another draw of subjects.

Subjects are resampled with replacement, whole: every row of a drawn subject comes along, so
repeated calls of one subject do not pass for independent subjects.
"""
from typing import Callable

import numpy as np
import pandas as pd

LOW, HIGH = "low", "high"


def interval(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], pd.Series],
    draws: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> pd.DataFrame:
    """
    The percentile bootstrap interval of each entry of `statistic(frame)`.

    Args:
        frame: rows indexed by subject; a subject may hold several rows.
        statistic: turns such a frame into one series.
        draws: resamples.
        seed: for the resampling.
        level: coverage of the interval.

    Returns:
        A frame indexed like the statistic, with the columns `low` and `high`.
    """
    rng = np.random.default_rng(seed)
    rows = list(pd.Series(np.arange(len(frame))).groupby(frame.index.to_numpy()).indices.values())
    drawn = []
    for _ in range(draws):
        picked = rng.integers(0, len(rows), len(rows))
        drawn.append(statistic(frame.iloc[np.concatenate([rows[i] for i in picked])]))
    edges = pd.DataFrame(drawn).quantile([(1 - level) / 2, (1 + level) / 2])
    edges.index = [LOW, HIGH]
    return edges.T
