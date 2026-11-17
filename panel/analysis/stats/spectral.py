"""
The singular value decomposition of a row-centred matrix, and how many directions it has.

Rows are outcomes, columns are profiles: centring each row on its mean over the profiles
leaves how each profile departs from the typical answer. A left singular vector is then an
axis over the outcomes - which rise as which fall - and the right singular vectors hold the
profiles' scores along it.
"""
from typing import Dict, Tuple

import numpy as np

SUM_RATIO, STABLE_RANK, EFFECTIVE_AXES = "sum s / s1", "stable rank", "effective number of axes"


def centred_svd(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    `U, s, Vt` of the matrix with each row's mean taken off; each axis's sign set so the
    largest entry of its `U` column is positive.
    """
    values = np.asarray(matrix, dtype=float)
    left, singular, right = np.linalg.svd(values - values.mean(axis=1, keepdims=True), full_matrices=False)
    sign = np.sign(left[np.abs(left).argmax(axis=0), range(left.shape[1])])
    sign[sign == 0] = 1.0
    return left * sign, singular, right * sign[:, None]


def soft_ranks(singular: np.ndarray) -> Dict[str, float]:
    """Three ways of counting the directions a matrix has, each 1 for one direction."""
    singular = np.asarray(singular, dtype=float)
    if singular[0] <= 0:
        raise ValueError("the matrix has no direction: every singular value is zero")
    shares = singular ** 2 / (singular ** 2).sum()
    held = shares[shares > 1e-15]
    return {
        SUM_RATIO: float(singular.sum() / singular[0]),
        STABLE_RANK: float((singular ** 2).sum() / singular[0] ** 2),
        EFFECTIVE_AXES: float(np.exp(-(held @ np.log(held)))),
    }
