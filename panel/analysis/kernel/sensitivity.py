"""
How far a shift in the population can move what it chooses (`stats.sensitivity`).

- any shift: the largest gain and loss of each outcome when a share `ε` of the population is
  replaced by whoever moves it most - exact; for a small `ε` it is `ε` times the outcome's
  range over the profiles, so it says little beyond that range;
- one factor: the largest gain of each outcome when only one factor's distribution shifts by
  `ε`, the rest of the make-up kept within each level; this is the part that says which
  factor a population shift would have to move.

`q0` is the run's own population, or every profile once. A level held by fewer profiles than
`least` is left out of the one-factor shift, which a few profiles would set by their noise.
"""
from typing import List, Tuple

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..output import Output
from ..plots.matrices import grid
from ..stats.sensitivity import sensitivity as largest_gain
from .matrix import Kernel, KernelSet

NAME, FACTORS = "kernel_sensitivity", "kernel_sensitivity_factors"


def one_factor(kernel: Kernel, population: pd.Series, share: float, least: int) -> pd.DataFrame:
    """Per outcome and factor, the largest gain when only that factor's distribution shifts by `share`."""
    found = {}
    for column in kernel.factors.columns:
        levels = kernel.grouping(column)
        counts = levels.value_counts()
        kept = levels.isin(counts.index[counts >= least])
        weights = population[kept]
        mass = weights.groupby(levels[kept]).sum()
        if len(mass) < 2:
            continue
        means = kernel.matrix.loc[:, kept].mul(weights, axis=1).T.groupby(levels[kept]).sum().div(mass, axis=0)
        mass = mass / mass.sum()
        found[column] = {outcome: float(largest_gain(means[outcome].to_numpy(), mass.to_numpy(), share)[0]) for outcome in kernel.outcomes}
    return pd.DataFrame(found)


def sensitivity(kernels: KernelSet, out: Output, share: float = 0.1, uniform: bool = False, least: int = 30) -> Tuple[pd.DataFrame, List[Figure]]:
    """
    Per kernel and outcome, the largest gain and loss under any shift of `share`, and under a
    shift of one factor.

    Args:
        kernels: the kernels.
        out: takes the tables and the figures.
        share: `ε`, the share of the population that moves.
        uniform: start from every profile once instead of the run's own population.
        least: profiles a level needs to take part in a one-factor shift.

    Returns:
        The table - indexed by kernel and outcome: the largest gain, the largest loss, `ε` times
        the range, the largest one-factor gain and its factor, all in shares - with every
        factor's gains beside it (`kernel_sensitivity_factors`), and a figure per kernel: the
        one-factor gains, outcomes by factors, in percentage points.
    """
    rows, factors = {}, {}
    for label, kernel in kernels.kernels.items():
        population = pd.Series(1.0, index=kernel.matrix.columns) if uniform else kernel.weights.reindex(kernel.matrix.columns)
        population = population / population.sum()
        values = kernel.matrix.to_numpy()
        gains = one_factor(kernel, population, share, least)
        rows[label] = pd.DataFrame({
            "gain": [float(largest_gain(row, population.to_numpy(), share)[0]) for row in values],
            "loss": [float(largest_gain(-row, population.to_numpy(), share)[0]) for row in values],
            "ε x range": share * (values.max(axis=1) - values.min(axis=1)),
            "one-factor gain": gains.max(axis=1).reindex(kernel.outcomes).to_numpy() if len(gains.columns) else np.nan,
            "by": gains.idxmax(axis=1).reindex(kernel.outcomes).to_numpy() if len(gains.columns) else None,
        }, index=kernel.outcomes)
        factors[label] = gains
    table = out.table(NAME, pd.concat(rows, names=["kernel", "outcome"]))
    out.table(FACTORS, pd.concat(factors, names=["kernel", "outcome"]))
    start = "every profile once" if uniform else "the run's own population"
    figures = [
        out.figure(f"{FACTORS}_{number}", grid(frame.rename(index=kernels.labels), decimals=1, scale=100, cmap="Oranges",
                                              title=f"{label}: what shifting one factor by {share:g} can gain each outcome",
                                              note=f"largest gain, percentage points, when one factor's distribution moves by TV {share:g} from {start}"))
        for number, (label, frame) in enumerate(factors.items(), start=1) if len(frame.columns)
    ]
    return table, figures
