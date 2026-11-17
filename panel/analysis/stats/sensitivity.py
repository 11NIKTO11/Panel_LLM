"""
How far a population-weighted mean can move when part of the population is replaced.

A population is a distribution `q0` over profiles and `r` a value per profile; the mean is
`r · q0`. Replacing a share `ε` of the population means any `q` with `TV(q, q0) ≤ ε`, and the
largest mean among them takes `ε` of mass from the profiles with the lowest `r` and gives it
to the profile with the highest - which is exact, not a search.
"""
import numpy as np


def sensitivity(values: np.ndarray, population: np.ndarray, shares) -> np.ndarray:
    """One increase per `ε` of `shares`, one or several, at most `ε (max - min)`; `population` sums to 1."""
    values, population = np.asarray(values, dtype=float), np.asarray(population, dtype=float)
    order = np.argsort(values)
    weights = population[order]
    before = np.cumsum(weights) - weights
    shares = np.atleast_1d(np.asarray(shares, dtype=float))[:, None]
    moved = np.clip(shares - before, 0.0, weights)
    return shares[:, 0] * values.max() - (moved * values[order]).sum(axis=1)
