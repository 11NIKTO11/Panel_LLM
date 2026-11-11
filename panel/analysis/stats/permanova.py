"""
PERMANOVA: how much of the distances between subjects' compositions a predictor accounts for,
tested by permutation (Anderson 2001).

Nothing is assumed about the shape of the data, only a distance between two subjects. Each
predictor is tested alone here, not given the others, so two related predictors both get
credit for what they share: read `R2` as "alone it accounts for this much".

Distances, all between two rows of shares:

- `hellinger`: Euclidean distance of the square roots, over sqrt(2); at home with zeros.
- `aitchison`: Euclidean distance of the ILR coordinates, zeros replaced first.
- `js`: Jensen-Shannon distance.
- `l1`: sum of absolute differences.
- `correlation`: sqrt(2(1 - r)), r being Pearson's correlation of the two rows over the parts.
"""
from typing import Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform

from .compositional import DELTA, ilr

R2, F, P, DF = "R2", "F", "p", "df"
DISTANCES = ("hellinger", "aitchison", "js", "l1", "correlation")


def pairwise(shares: np.ndarray, distance: str, delta: float = DELTA) -> np.ndarray:
    """
    The n x n matrix of a named distance between the rows of `shares`, each rescaled to sum to 1.

    Raises:
        ValueError: on an unknown distance.
    """
    values = np.asarray(shares, dtype=float)
    values = values / values.sum(axis=1, keepdims=True)
    if distance == "hellinger":
        return squareform(pdist(np.sqrt(values))) / np.sqrt(2.0)
    if distance == "aitchison":
        return squareform(pdist(ilr(values, delta)))
    if distance == "js":
        return squareform(pdist(values, "jensenshannon"))
    if distance == "l1":
        return squareform(pdist(values, "cityblock"))
    if distance == "correlation":
        return np.sqrt(2.0 * squareform(pdist(values, "correlation")))
    raise ValueError(f"distance must be one of {DISTANCES}, not {distance!r}")


def gower(distances: np.ndarray) -> np.ndarray:
    """The Gower-centred matrix of squared distances, whose trace is the total sum of squares."""
    n = len(distances)
    centring = np.eye(n) - 1.0 / n
    return -0.5 * centring @ (distances ** 2) @ centring


def permanova(centred: np.ndarray, columns: np.ndarray, permutations: int = 999, seed: int = 0) -> Tuple[int, float, float, float]:
    """
    One predictor's test on a Gower-centred matrix.

    Args:
        centred: from `gower`.
        columns: from `design.block`, the same rows.
        permutations: for the p-value.
        seed: of the permutations.

    Returns:
        Degrees of freedom, R2 (effect over total sum of squares), pseudo-F, and the p-value:
        the share of permutations, the observed order counted in, whose F is at least the
        observed one.

    Raises:
        ValueError: if the predictor leaves no residual to test against.
    """
    n = len(centred)
    full = np.column_stack([np.ones(n), columns])
    hat = full @ np.linalg.pinv(full)
    degrees = int(round(np.trace(hat))) - 1
    total = float(np.trace(centred))
    if degrees < 1 or n - 1 - degrees < 1:
        raise ValueError(f"a predictor with {degrees} degrees of freedom over {n} subjects leaves nothing to test")

    def ratio(matrix: np.ndarray) -> float:
        effect = float(np.sum(hat * matrix))
        return (effect / degrees) / ((total - effect) / (n - 1 - degrees))

    observed = ratio(centred)
    rng = np.random.default_rng(seed)
    at_least = 0
    for _ in range(permutations):
        order = rng.permutation(n)
        at_least += ratio(centred[np.ix_(order, order)]) >= observed
    return degrees, float(np.sum(hat * centred)) / total, observed, (at_least + 1) / (permutations + 1)
