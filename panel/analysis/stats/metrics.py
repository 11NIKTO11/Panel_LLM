"""
How far two distributions over the same options are from each other.

- MAE, RMSE: mean absolute and root mean squared difference of the shares.
- Correlation: Pearson, over the options.
- JS divergence: Jensen-Shannon, natural log; finite with empty options, symmetric.
- KL divergence: of the source from the reference. It is infinite when the source has an
  empty option the reference fills, so both get `FLOOR` added to every option first - the
  same floor for every source, or two sources' values could not be compared.

Options are names without an order, so nothing here depends on the order they come in.
"""
from typing import Callable, Dict, Sequence

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

MAE, RMSE, CORRELATION, JS, KL = "MAE", "RMSE", "Correlation", "JS divergence", "KL divergence"
METRICS = (MAE, RMSE, CORRELATION, JS, KL)
FLOOR = 1e-6


def _correlation(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.corrcoef(p, q)[0, 1]) if np.std(p) and np.std(q) else np.nan


MEASURES: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    MAE: lambda p, q: float(np.mean(np.abs(p - q))),
    RMSE: lambda p, q: float(np.sqrt(np.mean((p - q) ** 2))),
    CORRELATION: _correlation,
    JS: lambda p, q: float(jensenshannon(p, q) ** 2),
    KL: lambda p, q: float(entropy(p + FLOOR, q + FLOOR)),
}


def distances(reference: Sequence[float], source: Sequence[float], metrics: Sequence[str] = METRICS) -> Dict[str, float]:
    """
    The named metrics (all by default) between a source and the reference it is held against.

    Both are rescaled to sum to 1 first. A distribution that sums to zero (nobody chose
    anything) has no distance to anything: every metric is NaN.

    Raises:
        ValueError: on a negative entry.
    """
    p, q = np.asarray(reference, dtype=float), np.asarray(source, dtype=float)
    if (p < 0).any() or (q < 0).any():
        raise ValueError("a distribution has a negative entry")
    if p.sum() == 0 or q.sum() == 0:
        return dict.fromkeys(metrics, np.nan)
    p, q = p / p.sum(), q / q.sum()
    return {metric: MEASURES[metric](p, q) for metric in metrics}
