"""
Replicates of a statistic, and the interval they give.

- half-samples: a random half of the rows, without replacement. With half the rows the spread
  of a statistic around its full-sample value is about its sampling spread (the finite
  population correction is 1/2), so the percentile interval of the replicates is about a 95 %
  interval. Drawing with replacement instead would duplicate rows, which inflates a
  bias-corrected share like ε² most for factors with many levels.
- a statistic that is biased on half the rows (an out-of-sample R² is lower there) keeps only
  the spread of its replicates, re-centred on the full estimate.
"""
from typing import List

import numpy as np

LEVEL = 0.95


def halves(count: int, draws: int, seed: int = 0) -> List[np.ndarray]:
    """`draws` random halves of `count` rows, each sorted positions without repeats."""
    random = np.random.default_rng(seed)
    return [np.sort(random.choice(count, count // 2, replace=False)) for _ in range(draws)]


def interval(replicates: np.ndarray, estimate: float, recentre: bool = False, level: float = LEVEL) -> tuple:
    """
    (low, high) of the replicates' percentile interval; with `recentre` the replicates' spread
    around their mean laid around the estimate. NaN replicates are left out; all NaN gives (NaN, NaN).
    """
    found = np.asarray(replicates, dtype=float)
    found = found[~np.isnan(found)]
    if not len(found):
        return float("nan"), float("nan")
    low, high = np.quantile(found, [(1 - level) / 2, (1 + level) / 2])
    if recentre:
        middle = found.mean()
        return float(estimate - (middle - low)), float(estimate + (high - middle))
    return float(low), float(high)
