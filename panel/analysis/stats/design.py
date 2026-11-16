"""
A predictor as columns of a design matrix, without the intercept:

- numeric: itself, one column;
- holding several answers per cell: an indicator per answer, without the commonest one;
- anything else: an indicator per level, without the first.

One is left out each time because, with the intercept, it is what the others are read against.

A penalised model needs no level left out: `indicators` keeps every level, a missing answer
being one more; `with_interactions` adds the products of every two predictors' indicators.
A numeric predictor that should enter by bands comes in already cut, as a categorical.
"""
import itertools
from typing import Sequence, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def block(values: pd.Series, several: Union[str, None] = None) -> np.ndarray:
    """`values` has no missing values; `several` is the separator of a cell holding several answers."""
    if is_numeric_dtype(values):
        columns = values.to_numpy(dtype=float)[:, None]
    elif several:
        held = values.str.get_dummies(several).astype(float)
        columns = held.drop(columns=held.sum().idxmax()).to_numpy()
    else:
        columns = pd.get_dummies(values, drop_first=True, dtype=float).to_numpy()
    columns = columns[:, columns.std(axis=0) > 0]
    if not columns.shape[1]:
        raise ValueError(f"predictor '{values.name}' does not vary")
    return columns


def indicators(values: pd.Series, several: Union[str, None] = None, missing: str = "(missing)") -> np.ndarray:
    """
    A numeric predictor stays one column; a categorical gets a column per category, in their
    order, held or not; in any other a missing value is the level `missing`.
    """
    categorical = isinstance(values.dtype, pd.CategoricalDtype)
    if categorical or is_numeric_dtype(values):
        if values.isna().any():
            raise ValueError(f"numeric predictor '{values.name}' has missing values")
        return pd.get_dummies(values, dtype=float).to_numpy() if categorical else values.to_numpy(dtype=float)[:, None]
    values = values.fillna(missing)
    return (values.str.get_dummies(several) if several else pd.get_dummies(values)).to_numpy(dtype=float)


def with_interactions(blocks: Sequence[np.ndarray]) -> np.ndarray:
    """The blocks side by side, then every pair's column products, without those that are zero in every row."""
    products = [(first[:, :, None] * second[:, None, :]).reshape(len(first), -1) for first, second in itertools.combinations(blocks, 2)]
    joined = np.hstack(products) if products else np.empty((len(blocks[0]), 0))
    return np.hstack([*blocks, joined[:, joined.any(axis=0)]])
