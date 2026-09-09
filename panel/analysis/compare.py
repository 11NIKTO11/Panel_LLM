"""
Distances between distributions, and tables of them: one source against a reference, or every pair.

The metrics are the colleague's (`calculate_distances` in the former panel_analysis.ipynb),
extracted, not re-derived: MAE, RMSE, Pearson correlation, KL divergence, Jensen-Shannon
divergence and Wasserstein distance over positions. One deviation is recorded: the notebook
had stubbed Wasserstein to a constant 1 while its position-weighted version sat unused
beside it; that version is what runs here.
"""
from itertools import combinations
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, wasserstein_distance

METRICS = ("MAE", "RMSE", "Correlation", "KL Divergence", "Jensen-Shannon Divergence", "Wasserstein Distance")
EPS = 1e-12


def distances(p: Sequence[float], q: Sequence[float]) -> Dict[str, float]:
    """
    Every metric between two distributions over the same positions; `p` is the reference for KL.

    Both are normalised to sum to 1 first; divergences use a floor of 1e-12 for zeros.

    Raises:
        ValueError: if either has a negative entry or sums to zero.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if (p < 0).any() or (q < 0).any() or p.sum() <= 0 or q.sum() <= 0:
        raise ValueError("a distribution has a negative entry or sums to zero")
    p, q = p / p.sum(), q / q.sum()
    p_safe, q_safe = np.clip(p, EPS, 1.0), np.clip(q, EPS, 1.0)
    p_safe, q_safe = p_safe / p_safe.sum(), q_safe / q_safe.sum()
    positions = np.arange(len(p))
    return {
        "MAE": float(np.mean(np.abs(p - q))),
        "RMSE": float(np.sqrt(np.mean((p - q) ** 2))),
        "Correlation": float(np.corrcoef(p, q)[0, 1]) if np.std(p) and np.std(q) else np.nan,
        "KL Divergence": float(entropy(p_safe, q_safe)),
        "Jensen-Shannon Divergence": float(jensenshannon(p_safe, q_safe) ** 2),
        "Wasserstein Distance": float(wasserstein_distance(positions, positions, p, q)),
    }


def distance_table(reference: pd.DataFrame, other: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """
    Every metric per row between two frames of distributions indexed by group.

    Raises:
        ValueError: if the frames do not share an index, or a row is not a distribution.
    """
    if not reference.index.equals(other.index):
        raise ValueError("the frames must share an index")
    columns = list(columns)
    rows = {group: distances(reference.loc[group, columns], other.loc[group, columns]) for group in reference.index}
    return pd.DataFrame(rows).T.reindex(columns=list(METRICS))


def smooth(frame: pd.DataFrame, columns: Sequence[str], eps: float = 1e-6) -> pd.DataFrame:
    """The frame with `eps` added to every entry of `columns` and those rows rescaled to 1, so KL stays finite."""
    out = frame.copy()
    values = out[list(columns)].fillna(0.0).astype(float) + eps
    out[list(columns)] = values.div(values.sum(axis=1), axis=0)
    return out


def pairwise(
    sources: Dict[str, pd.DataFrame],
    columns: Sequence[str],
    pairs: Union[List[Tuple[str, str]], None] = None,
    metric: str = "MAE",
    decimals: Union[int, None] = None,
) -> pd.DataFrame:
    """
    One metric between every pair of sources, per group: a table indexed by group with a column `A vs B` per pair.

    Args:
        sources: name -> distributions indexed by group, all on the same index.
        columns: the distribution columns compared.
        pairs: which pairs, in which order; every combination in source order by default.
        metric: one of `METRICS`.
        decimals: round the result, as the former notebook did, when given.

    Raises:
        ValueError: on an unknown metric or a pair naming an unknown source.
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, not {metric!r}")
    pairs = pairs or list(combinations(sources, 2))
    table = pd.DataFrame(index=next(iter(sources.values())).index)
    for first, second in pairs:
        table[f"{first} vs {second}"] = distance_table(sources[first], sources[second], columns)[metric]
    return table.round(decimals) if decimals is not None else table
