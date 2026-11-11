"""
Compositions - rows of shares summing to 1 - made fit for statistics that assume free real
coordinates: zero replacement, then the isometric log-ratio (ILR) transform.

A log-ratio has no value at zero, so zeros are replaced first, by something of the size a
share could have gone unnoticed at: `DELTA`, half the 0.01 step answers are usually given in.
The replacement is multiplicative - the other shares shrink in proportion, so the row still
sums to 1 and their ratios to each other do not change. Clipping at 1e-12 instead puts a
zero some 25 log-units from everything else, and those few cells then decide every test.
"""
import numpy as np

DELTA = 0.005


def helmert_basis(k: int) -> np.ndarray:
    """
    Orthonormal Helmert sub-matrix of shape (k, k-1), a basis of the subspace orthogonal to ones.

    Raises:
        ValueError: if k < 2.
    """
    if k < 2:
        raise ValueError("k must be >= 2 for an ILR transform")
    basis = np.zeros((k, k - 1), dtype=float)
    for j in range(1, k):
        denominator = np.sqrt(j * (j + 1))
        basis[:j, j - 1] = 1.0 / denominator
        basis[j, j - 1] = -j / denominator
    return basis


def replace_zeros(compositions: np.ndarray, delta: float = DELTA) -> np.ndarray:
    """
    Each row rescaled to sum to 1 with its zeros set to `delta` and the rest shrunk to make room.

    Raises:
        ValueError: on a negative entry, a row summing to zero, or a `delta` that leaves no
            room for the non-zero parts.
    """
    values = np.asarray(compositions, dtype=float)
    if (values < 0).any() or (values.sum(axis=1) <= 0).any():
        raise ValueError("a composition has a negative entry or sums to zero")
    if not 0 < delta * values.shape[1] < 1:
        raise ValueError(f"delta {delta} leaves no room among {values.shape[1]} parts")
    values = values / values.sum(axis=1, keepdims=True)
    zeros = values <= 0
    return np.where(zeros, delta, values * (1 - delta * zeros.sum(axis=1, keepdims=True)))


def ilr(compositions: np.ndarray, delta: float = DELTA) -> np.ndarray:
    """ILR coordinates of each row, k-1 of them for k parts, after `replace_zeros`."""
    values = replace_zeros(compositions, delta)
    return np.log(values) @ helmert_basis(values.shape[1])
