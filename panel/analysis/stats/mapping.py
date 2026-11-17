"""
A distance matrix drawn as a map and held against another map of the same things.

- `mds`: metric multidimensional scaling to two dimensions, with Kruskal's stress-1 - how
  badly the map's distances miss the given ones; below 0.1 is good.
- `procrustes`: one map shifted, rotated or mirrored and rescaled onto a target, and the
  correlation r of the fit, 1 for a perfect match. With few points such a fit is cheap, so
  `procrustes_test` asks how many relabellings of the points fit at least as well.
- `mantel`: Spearman correlation of two distance matrices, pair by pair, tested the same way.

The tests go through every relabelling, so they are exact and meant for a handful of things.
"""
from itertools import permutations
from typing import Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.manifold import MDS

MOST = 8


def mds(distances: np.ndarray, seed: int = 0) -> Tuple[np.ndarray, float]:
    """Two coordinates per row of a distance matrix, and the map's stress-1."""
    given = np.asarray(distances, dtype=float)
    points = MDS(
        n_components=2, metric_mds=True, metric="precomputed", init="random", n_init=16, max_iter=3000, random_state=seed, normalized_stress=False,
    ).fit_transform(given)
    upper = np.triu_indices(len(given), 1)
    drawn = np.linalg.norm(points[:, None] - points[None], axis=-1)
    return points, float(np.sqrt(((given[upper] - drawn[upper]) ** 2).sum() / (given[upper] ** 2).sum()))


def procrustes(target: np.ndarray, points: np.ndarray) -> Tuple[np.ndarray, float]:
    """The points laid onto the target, in the target's units, and the correlation of the fit."""
    centre = target.mean(axis=0)
    wanted, moved = target - centre, points - points.mean(axis=0)
    left, singular, right = np.linalg.svd(moved.T @ wanted)
    fitted = singular.sum() / (moved ** 2).sum() * moved @ left @ right + centre
    return fitted, float(singular.sum() / np.sqrt((wanted ** 2).sum() * (moved ** 2).sum()))


def _orders(count: int):
    if count > MOST:
        raise ValueError(f"an exact test over every relabelling is meant for at most {MOST} things, not {count}")
    return permutations(range(count))


def procrustes_test(target: np.ndarray, points: np.ndarray) -> Tuple[float, float]:
    """The Procrustes correlation and the share of relabellings of the points that fit at least as well."""
    observed = procrustes(target, points)[1]
    fits = np.array([procrustes(target[list(order)], points)[1] for order in _orders(len(points))])
    return observed, float((fits >= observed - 1e-12).mean())


def mantel(first: np.ndarray, second: np.ndarray) -> Tuple[float, float]:
    """Spearman correlation of two distance matrices' pairs, and the share of relabellings at least as correlated."""
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    upper = np.triu_indices(len(first), 1)
    observed = spearmanr(first[upper], second[upper])[0]
    found = np.array([spearmanr(first[upper], second[np.ix_(order, order)][upper])[0] for order in _orders(len(first))])
    return float(observed), float((found >= observed - 1e-12).mean())
