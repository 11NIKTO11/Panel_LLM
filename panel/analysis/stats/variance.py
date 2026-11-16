"""
How much of the variation of answers over profiles the profiles' factors account for.

`Y` is profiles x outcomes. Three measures:

- `eps2`: one factor alone, by its level means, with the chance part taken off - a level mean
  of few profiles follows the noise, so a factor with many levels explains some variance even
  when it does nothing (epsilon squared, the bias-corrected eta squared).
- `cv_r2` / `loo_r2`: a ridge regression on a design matrix, scored out of sample, so that
  adding columns cannot explain variance by fitting the noise.
- `factorial`: the exact split of a complete, balanced design into main effects and two-way
  interactions, by orthogonal projections on level and cell means.

Every share is of the summed variance over the outcomes ("pooled") and per outcome.
"""
import itertools
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict

CV_ALPHAS = np.logspace(-1, 4, 11)
LOO_ALPHAS = np.logspace(-2, 4, 13)


def _matrix(values: np.ndarray, rows: int) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(rows, -1)


def eps2(values: np.ndarray, groups: Sequence) -> Tuple[float, np.ndarray]:
    """(pooled, one per column) of `values`, n x k or n; NaN where a column does not vary."""
    labels = np.asarray(groups)
    values = _matrix(values, len(labels))
    means = pd.DataFrame(values).groupby(labels).transform("mean").to_numpy()
    levels, count = len(pd.unique(labels)), len(labels)
    total = ((values - values.mean(axis=0)) ** 2).sum(axis=0)
    between = ((means - values.mean(axis=0)) ** 2).sum(axis=0)
    within = total - between
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled = (between.sum() - (levels - 1) * within.sum() / (count - levels)) / total.sum()
        each = (between - (levels - 1) * within / (count - levels)) / total
    return float(pooled), each


def cv_r2(features: np.ndarray, values: np.ndarray, folds: int = 5, seed: int = 0) -> Tuple[np.ndarray, float]:
    """(one per column, pooled) - the other way round from `eps2`; the penalty is chosen inside each fold."""
    values = _matrix(values, len(features))
    predicted = cross_val_predict(RidgeCV(alphas=CV_ALPHAS, gcv_mode="eigen"), features, values, cv=KFold(folds, shuffle=True, random_state=seed))
    total = ((values - values.mean(axis=0)) ** 2).sum(axis=0)
    left = ((values - predicted) ** 2).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 1 - left / total, float(1 - left.sum() / total.sum())


def loo_r2(features: np.ndarray, values: np.ndarray) -> float:
    """Leave-one-out R² of a ridge regression, pooled over the outcomes; the penalty is the one with the least leave-one-out error."""
    values = _matrix(values, len(features))
    fitted = RidgeCV(alphas=LOO_ALPHAS, store_cv_results=True).fit(features, values)
    errors = fitted.cv_results_.reshape(len(features), -1, len(LOO_ALPHAS))
    return float(1 - errors[:, :, errors.mean(axis=(0, 1)).argmin()].sum() / ((values - values.mean(axis=0)) ** 2).sum())


def factorial(values: np.ndarray, factors: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    name -> sum of squares per outcome: one entry per factor, one per pair named `a x b`, `rest`
    (the higher interactions) and `total`, which the others add up to.
    """
    values = _matrix(values, len(factors))
    sizes = [factors[column].nunique() for column in factors.columns]
    if len(factors) != int(np.prod(sizes)) or factors.duplicated().any():
        raise ValueError(f"not a complete factorial: {len(factors)} rows for levels {dict(zip(factors.columns, sizes))}")
    centred = values - values.mean(axis=0)
    frame = pd.DataFrame(centred)

    def projection(columns) -> np.ndarray:
        keys = [factors[column].to_numpy() for column in columns]
        return frame.groupby(keys).transform("mean").to_numpy()

    mains = {column: projection([column]) for column in factors.columns}
    split = {column: (effect ** 2).sum(axis=0) for column, effect in mains.items()}
    explained = sum(mains.values())
    for first, second in itertools.combinations(factors.columns, 2):
        effect = projection([first, second]) - mains[first] - mains[second]
        split[f"{first} x {second}"] = (effect ** 2).sum(axis=0)
        explained = explained + effect
    split["rest"] = ((centred - explained) ** 2).sum(axis=0)
    split["total"] = (centred ** 2).sum(axis=0)
    return split
