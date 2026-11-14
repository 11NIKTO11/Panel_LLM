"""
MANOVA of per-subject compositions on the subjects' own columns, via statsmodels.

The dependent side is the ILR coordinates of the shares, so the sum-to-1 constraint stays out
of the model. All predictors enter together, each as its block of design columns
(`design.block`), and each is tested given the others: the hypothesis is that its whole block
has no effect.

Pillai's trace grows with the number of columns a predictor has, so two predictors' traces do
not compare. What does is the trace divided by its largest possible value, `s = min(k-1,
hypothesis df)`: the multivariate partial eta squared, the share of the variation left over
by the other predictors that this one accounts for, between 0 and 1.

Two parts leave one coordinate, and MANOVA on one variable is ANOVA, which statsmodels'
`MANOVA` refuses: the same hypotheses are then F-tested on the least-squares fit, and the trace
is the partial R squared, `qF / (qF + residual df)` - what Pillai's trace is for one variable.
"""
from typing import Dict, List, Mapping, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from statsmodels.multivariate.manova import MANOVA
from statsmodels.regression.linear_model import OLS

from .compositional import DELTA, ilr
from .design import block

PILLAI, ETA, F, P, DF = "Pillai", "partial eta2", "F", "p", "df"
STATISTIC = "Pillai's trace"


def manova(
    subjects: pd.DataFrame, shares: pd.DataFrame, predictors: Sequence[str],
    several: Union[Mapping[str, str], None] = None, delta: float = DELTA,
) -> pd.DataFrame:
    """
    Each predictor's multivariate test, given the other predictors.

    Args:
        subjects: one row per subject with the predictor columns, no missing values.
        shares: the same subjects' compositions, a column per part.
        predictors: columns of `subjects`.
        several: predictor -> the separator of its cells holding several answers.
        delta: what a zero share is replaced by before the log-ratio transform.

    Returns:
        One row per predictor: `df` (hypothesis degrees of freedom), `Pillai`, `partial
        eta2`, `F` and `p`.

    Raises:
        ValueError: if the two frames are not the same subjects, a predictor does not vary,
            or there are too few subjects for the design.
    """
    if not subjects.index.equals(shares.index):
        raise ValueError("subjects and shares are not the same rows")
    several = several or {}
    coordinates = ilr(shares.to_numpy(dtype=float), delta)
    blocks = {column: block(subjects[column], several.get(column)) for column in predictors}
    design = np.column_stack([np.ones(len(subjects)), *blocks.values()])
    if len(subjects) <= design.shape[1] + coordinates.shape[1]:
        raise ValueError(f"{len(subjects)} subjects are too few for {design.shape[1]} design columns and {coordinates.shape[1]} coordinates")

    hypotheses, start = [], 1
    for column, columns in blocks.items():
        contrast = np.zeros((columns.shape[1], design.shape[1]))
        contrast[np.arange(columns.shape[1]), start + np.arange(columns.shape[1])] = 1.0
        hypotheses.append((column, contrast, None))
        start += columns.shape[1]
    tests = _tests(coordinates, design, hypotheses)

    rows = {}
    for column, columns in blocks.items():
        value, ratio, chance = tests[column]
        rows[column] = {DF: columns.shape[1], PILLAI: value, ETA: value / min(coordinates.shape[1], columns.shape[1]), F: ratio, P: chance}
    return pd.DataFrame(rows).T.astype({DF: int})


def _tests(coordinates: np.ndarray, design: np.ndarray, hypotheses: List[tuple]) -> Dict[str, Tuple[float, float, float]]:
    """Hypothesis -> Pillai's trace, its F and p; by ANOVA for a single coordinate (module docstring)."""
    if coordinates.shape[1] > 1:
        tests = MANOVA(coordinates, design).mv_test(hypotheses)
        found = {name: tests.results[name]["stat"].loc[STATISTIC] for name, _, _ in hypotheses}
        return {name: (float(row["Value"]), float(row["F Value"]), float(row["Pr > F"])) for name, row in found.items()}
    fitted = OLS(coordinates[:, 0], design).fit()
    out = {}
    for name, contrast, _ in hypotheses:
        test = fitted.f_test(contrast)
        ratio, q = float(np.squeeze(test.fvalue)), contrast.shape[0]
        out[name] = (q * ratio / (q * ratio + fitted.df_resid), ratio, float(test.pvalue))
    return out
