"""
What each factor explains alone: ε², the share of an outcome's variance over the profiles that
the factor's level means account for, with the chance part taken off (`stats.variance.eps2`).

On a designed kernel the factors are independent, so the shares do not overlap and ε² is
exact; on a sampled one related factors (region and district, income and living standard)
each claim what they share, which `shapley` splits fairly. Beside it, the largest TV between
the mean answers of two of the factor's levels: how far the factor moves the answer at most.
"""
from typing import List, Tuple

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..output import Output
from ..plots.matrices import grid
from ..stats.distributions import tv
from ..stats.variance import eps2
from .matrix import KernelSet

NAME, OUTCOMES = "kernel_importance", "kernel_importance_outcomes"
EPS2, LEVELS, SPREAD = "ε²", "levels", "largest TV between level means"


def importance(kernels: KernelSet, out: Output, least: int = 30) -> Tuple[pd.DataFrame, List[Figure]]:
    """
    Per kernel and factor, ε² pooled over the outcomes and the largest TV between level means.

    Args:
        kernels: the kernels.
        out: takes the tables and the figures.
        least: levels held by fewer profiles are left out of the largest TV, which a level of a
            few profiles would set by their noise.

    Returns:
        The table - indexed by kernel and factor: ε², the number of levels, the largest TV - with
        ε² per outcome beside it (`kernel_importance_outcomes`), and the figures: pooled ε² of
        every factor by kernel, then per kernel ε² by factor and outcome.
    """
    rows, outcomes = {}, {}
    for label, kernel in kernels.kernels.items():
        values = kernel.matrix.to_numpy().T
        found, each = {}, {}
        for column in kernel.factors.columns:
            groups = kernel.grouping(column).to_numpy()
            pooled, per = eps2(values, groups)
            frame = pd.DataFrame(values).groupby(groups)
            means = frame.mean()[frame.size() >= least].to_numpy()
            found[column] = {EPS2: pooled, LEVELS: len(pd.unique(groups)), SPREAD: float(tv(means[:, None], means[None]).max()) if len(means) > 1 else np.nan}
            each[column] = pd.Series(per, index=kernel.outcomes)
        rows[label] = pd.DataFrame(found).T.sort_values(EPS2, ascending=False)
        outcomes[label] = pd.DataFrame(each).T
    table = out.table(NAME, pd.concat(rows, names=["kernel", "factor"]))
    out.table(OUTCOMES, pd.concat(outcomes, names=["kernel", "factor"]))
    pooled = pd.DataFrame({label: frame[EPS2] for label, frame in rows.items()})
    figures = [out.figure(NAME, grid(pooled.clip(lower=0), decimals=2, title=f"{kernels.topic.name}: what each factor explains alone",
                                     note="ε²: the bias-corrected share of the variance over the profiles, pooled over the outcomes; empty where a kernel has no such factor"))]
    for number, (label, frame) in enumerate(outcomes.items(), start=1):
        figures.append(out.figure(f"{OUTCOMES}_{number}", grid(frame.clip(lower=0).rename(columns=kernels.labels), decimals=2, title=f"{label}: what each factor explains of each outcome",
                                                            note="ε² per outcome, negative estimates shown as 0")))
    return table, figures
