"""
PERMANOVA: distance-based multivariate analysis of variance with a permutation test.

Anderson (2001) sums of squares over a Gower-centred distance matrix, with predictors
dummy-encoded; the distances themselves are computed here for several metrics. The colleague's
implementation from the former panel_analysis.ipynb, with its error swallowing removed.
"""
from dataclasses import dataclass
from typing import Dict, Literal, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .compositional import ilr

DistanceName = Literal["euclidean_ilr", "hellinger", "l1", "mae", "correlation", "js"]
DISTANCES = ("euclidean_ilr", "hellinger", "l1", "mae", "correlation", "js")


def pairwise_distances(points: np.ndarray, metric: str) -> np.ndarray:
    """
    The n x n distance matrix of the rows of `points`.

    Metrics: `euclidean_ilr` (plain Euclidean, on ILR coordinates by convention), `hellinger`,
    `l1`, `mae` (mean absolute difference), `correlation` (sqrt(2(1-r))) and `js`
    (Jensen-Shannon distance); the probability metrics clip and rescale rows first.

    Raises:
        ValueError: on an unknown metric.
    """
    metric = metric.lower()
    n = points.shape[0]
    if metric == "euclidean_ilr":
        difference = points[:, None, :] - points[None, :, :]
        return np.sqrt(np.sum(difference ** 2, axis=-1))
    if metric in ("hellinger", "js"):
        points = np.clip(points, 1e-12, None)
        points = points / points.sum(axis=1, keepdims=True)
    if metric == "hellinger":
        roots = np.sqrt(points)
        difference = roots[:, None, :] - roots[None, :, :]
        return np.sqrt(np.sum(difference ** 2, axis=-1)) / np.sqrt(2.0)
    if metric in ("l1", "mae"):
        difference = np.abs(points[:, None, :] - points[None, :, :])
        return np.sum(difference, axis=-1) if metric == "l1" else np.mean(difference, axis=-1)
    if metric == "correlation":
        centred = points - points.mean(axis=0, keepdims=True)
        gram = centred @ centred.T
        diagonal = np.sqrt(np.diag(gram))
        denominator = np.outer(diagonal, diagonal)
        with np.errstate(divide="ignore", invalid="ignore"):
            correlation = np.where(denominator > 0, gram / denominator, 0.0)
        return np.sqrt(2.0 * (1.0 - np.clip(correlation, -1.0, 1.0)))
    if metric == "js":
        distance = np.zeros((n, n), dtype=float)
        for i in range(n):
            p = points[i]
            for j in range(i + 1, n):
                q = points[j]
                m = 0.5 * (p + q)
                divergence = 0.5 * np.sum(p * (np.log(p) - np.log(m))) + 0.5 * np.sum(q * (np.log(q) - np.log(m)))
                distance[i, j] = distance[j, i] = float(np.sqrt(divergence))
        return distance
    raise ValueError(f"unsupported distance metric: {metric}")


@dataclass(frozen=True)
class PermanovaResult:
    """The test: degrees of freedom and sum of squares of the effect, its mean square, the F statistic and the permutation p-value."""
    df: int
    ss: float
    ms: float
    f_value: float
    p_value: float
    n_perm: int


def permanova(distance: np.ndarray, design: np.ndarray, n_perm: int = 999, seed: Union[int, None] = None) -> PermanovaResult:
    """
    PERMANOVA of a distance matrix on a numeric design matrix (n x p, no intercept).

    Sums of squares follow Anderson (2001) over the Gower-centred matrix; negatives from
    numerical noise are floored at zero, and the p-value counts permutations whose F is at
    least the observed one, plus one.
    """
    rng = np.random.default_rng(seed)
    n = distance.shape[0]
    centring = np.eye(n) - np.ones((n, n)) / n
    gower = -0.5 * centring @ (distance ** 2) @ centring
    identity = np.eye(n)

    def sums_of_squares(predictors: np.ndarray) -> Tuple[float, float]:
        with_intercept = np.column_stack([np.ones(n), predictors])
        hat = with_intercept @ np.linalg.pinv(with_intercept.T @ with_intercept) @ with_intercept.T
        return max(float(np.trace(hat @ gower)), 0.0), max(float(np.trace((identity - hat) @ gower)), 0.0)

    df_effect = design.shape[1]
    df_resid = max(n - 1 - df_effect, 1)

    def f_statistic(ss_effect: float, ss_resid: float) -> float:
        ms_effect = ss_effect / df_effect if df_effect > 0 else np.nan
        ms_resid = ss_resid / df_resid if df_resid > 0 else np.nan
        return ms_effect / ms_resid if ms_resid > 0 else np.nan

    ss_effect, ss_resid = sums_of_squares(design)
    f_observed = f_statistic(ss_effect, ss_resid)
    at_least = 0
    for _ in range(n_perm):
        f_permuted = f_statistic(*sums_of_squares(design[rng.permutation(n), :]))
        if not np.isnan(f_permuted) and f_permuted >= f_observed:
            at_least += 1
    return PermanovaResult(
        df=df_effect, ss=ss_effect, ms=ss_effect / df_effect if df_effect > 0 else np.nan,
        f_value=f_observed, p_value=(at_least + 1) / (n_perm + 1), n_perm=n_perm,
    )


def panel_permanova(
    profile: pd.DataFrame,
    results: pd.DataFrame,
    predictors: Sequence[str],
    columns: Sequence[str],
    distance: DistanceName = "euclidean_ilr",
    ilr_for_euclidean: bool = True,
    n_perm: int = 999,
    seed: Union[int, None] = None,
) -> Tuple[PermanovaResult, Dict[str, object]]:
    """
    PERMANOVA of per-subject distributions on profile columns.

    Args:
        profile: subjects with the predictor columns, indexed by id.
        results: one row per subject with the distribution columns, indexed by id.
        predictors: profile columns, dummy-encoded with the first level dropped.
        columns: the distribution columns of `results`.
        distance: see `pairwise_distances`.
        ilr_for_euclidean: transform to ILR coordinates before a Euclidean distance.
        n_perm: permutations for the p-value.
        seed: for the permutation RNG.

    Returns:
        The result and a report of what was tested (predictors, distance, n_obs, n_predictors).

    Raises:
        ValueError: on a predictor missing from the profile, or no complete rows.
    """
    missing = [column for column in predictors if column not in profile.columns]
    if missing:
        raise ValueError(f"predictors missing from the profile: {missing}")
    merged = profile.join(results, how="inner").copy()
    dependent = list(columns)
    for column in dependent:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    merged = merged.dropna(subset=dependent + list(predictors))
    if merged.empty:
        raise ValueError("no complete rows to test")

    values = merged[dependent].to_numpy(dtype=float)
    space = ilr(values) if distance == "euclidean_ilr" and ilr_for_euclidean else values
    matrix = pairwise_distances(space, distance)
    encoded = pd.concat(
        [pd.get_dummies(merged[column].astype("category") if merged[column].dtype == "object" else merged[column],
                        prefix=column, drop_first=True, dtype=float) for column in predictors],
        axis=1,
    ).to_numpy(dtype=float)
    result = permanova(matrix, encoded, n_perm=n_perm, seed=seed)
    report = {"predictors": list(predictors), "distance": distance, "n_obs": int(matrix.shape[0]), "n_predictors": int(encoded.shape[1])}
    return result, report
