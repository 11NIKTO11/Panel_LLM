"""
Distances between distributions over the same outcomes: total variation, `½ Σ |p - p'|`,
between 0 (the same) and 1 (no outcome in common).
"""
from typing import Tuple

import numpy as np


def tv(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Total variation between distributions along the last axis; broadcasts like `first - second`."""
    return 0.5 * np.abs(np.asarray(first, dtype=float) - np.asarray(second, dtype=float)).sum(axis=-1)


def pairwise_tv(rows: np.ndarray, chunk: int = 256) -> Tuple[np.ndarray, Tuple[float, int, int]]:
    """
    The n(n-1)/2 distances of the row pairs i < j (float32), and (largest, i, j).
    `chunk` rows are compared at a time, which bounds the memory to chunk x n x k.
    """
    rows = np.asarray(rows, dtype=float)
    count = len(rows)
    if count < 2:
        raise ValueError("pairwise distances need two rows or more")
    found, best = [], (-1.0, 0, 1)
    for start in range(0, count, chunk):
        distances = tv(rows[start:start + chunk, None], rows[None])
        upper = np.arange(count)[None] > np.arange(start, start + len(distances))[:, None]
        found.append(distances[upper].astype(np.float32))
        distances[~upper] = -1.0
        at = int(distances.argmax())
        if distances.flat[at] > best[0]:
            best = (float(distances.flat[at]), start + at // count, at % count)
    return np.concatenate(found), best
