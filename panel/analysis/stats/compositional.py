"""
Compositional data: the isometric log-ratio (ILR) transform over an orthonormal Helmert basis.

A row of option probabilities is a composition; ILR maps K parts onto K-1 unconstrained real
coordinates, which is what MANOVA and Euclidean distances need. The notebook defined the
basis twice (cells 8 and 13), identically; this is that definition.
"""
from typing import Sequence, Tuple, Union

import numpy as np
import pandas as pd


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


def ilr(compositions: np.ndarray, eps: float = 1e-12, renormalize: bool = True) -> np.ndarray:
    """
    ILR coordinates of each row: zeros clipped to `eps`, rows rescaled to 1, then log(X) @ basis.

    Raises:
        ValueError: if eps is not positive.
    """
    if eps <= 0:
        raise ValueError("eps must be > 0")
    values = np.clip(np.asarray(compositions, dtype=float), eps, None)
    if renormalize:
        values = values / np.clip(values.sum(axis=1, keepdims=True), eps, None)
    return np.log(values) @ helmert_basis(values.shape[1])


def ilr_frame(
    frame: pd.DataFrame,
    columns: Sequence[str],
    eps: float = 1e-12,
    renormalize: bool = True,
    prefix: str = "ilr_",
    return_basis: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, np.ndarray]]:
    """
    The frame with K-1 ILR columns added, `<prefix>1` .. `<prefix>K-1`, computed over `columns`.

    Raises:
        KeyError: if a column is missing.
    """
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"missing columns in frame: {missing}")
    coordinates = ilr(frame[list(columns)].to_numpy(dtype=float), eps, renormalize)
    out = frame.copy()
    out[[f"{prefix}{i + 1}" for i in range(coordinates.shape[1])]] = coordinates
    return (out, helmert_basis(len(columns))) if return_basis else out
