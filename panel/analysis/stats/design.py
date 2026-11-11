"""
A predictor as columns of a design matrix, without the intercept:

- numeric: itself, one column;
- holding several answers per cell: an indicator per answer, without the commonest one;
- anything else: an indicator per level, without the first.

One is left out each time because, with the intercept, it is what the others are read against.
"""
from typing import Union

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def block(values: pd.Series, several: Union[str, None] = None) -> np.ndarray:
    """
    The design columns of one predictor.

    Args:
        values: the predictor, no missing values.
        several: the separator of a cell holding several answers, or None.

    Raises:
        ValueError: if the predictor does not vary.
    """
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
