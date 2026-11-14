"""
What a population chooses by the kernel, `M q`, and what its make-up contributes.

- own: the run's own population, its subjects (a survey run's `q` is its sample; a grid's
  every cell once);
- every profile once: what the kernel says when no profile counts more than another;
- one factor evened out: the own population reweighted so that one factor's levels are equally
  common, the rest of its make-up kept within each level. The TV from the own answer is what
  that factor's make-up in the population contributes to the result.

Where the dataset has the subjects' own answers to the topic, the claimed choice stands beside
the own population's, in outcomes too (option shares times turnout, and not taking part).
"""
from typing import List, Tuple

import pandas as pd
from matplotlib.figure import Figure

from ..aggregate import claimed_distribution
from ..output import Output
from ..plots.matrices import grid
from ..stats.distributions import tv
from .matrix import Kernel, KernelSet, push

NAME, FACTORS = "kernel_populations", "kernel_populations_factors"
OWN, UNIFORM, CLAIMED = "own population", "every profile once", "claimed"


def claimed_outcomes(kernel: Kernel) -> pd.Series:
    """The subjects' own answers as outcomes, or an empty series when the dataset has none for the topic."""
    run = kernel.run
    if not run.claims:
        return pd.Series(dtype=float)
    claimed = claimed_distribution(run.subjects, run.claims, run.topic)
    options = list(run.topic.options)
    if not run.topic.turnout:
        return claimed[options]
    taking, staying = run.topic.turnout
    return pd.concat([claimed[options] * claimed[taking], pd.Series({staying: claimed[staying]})])[kernel.outcomes]


def evened(kernel: Kernel, column: str) -> pd.Series:
    """The own population with one factor's levels made equally common."""
    levels = kernel.grouping(column)
    mass = kernel.weights.groupby(levels).transform("sum")
    return kernel.weights / mass / levels.nunique()


def populations(kernels: KernelSet, out: Output) -> Tuple[pd.DataFrame, List[Figure]]:
    """
    Per kernel, `M q` for its own population, for every profile once, and the claimed choice;
    per kernel and factor, how far evening that factor out moves the answer.

    Returns:
        The table - indexed by kernel and population, a column per outcome, and the TV from
        the own population's answer - with the factors' table beside it
        (`kernel_populations_factors`: per kernel and factor, the TV and the largest change of
        an outcome, in shares), and the figure: the outcomes by population and kernel.
    """
    rows, factors = {}, {}
    for label, kernel in kernels.kernels.items():
        own = push(kernel, kernel.weights)
        found = {OWN: own, UNIFORM: push(kernel, pd.Series(1.0, index=kernel.matrix.columns))}
        claimed = claimed_outcomes(kernel)
        if len(claimed):
            found[CLAIMED] = claimed
        frame = pd.DataFrame(found).T
        frame["TV from own"] = [tv(row.to_numpy(), own.to_numpy()) for _, row in frame[kernel.outcomes].iterrows()]
        rows[label] = frame
        shifts = {}
        for column in kernel.factors.columns:
            moved = push(kernel, evened(kernel, column)) - own
            shifts[column] = {"TV": 0.5 * moved.abs().sum(), "largest change": moved.abs().max(), "outcome": moved.abs().idxmax()}
        factors[label] = pd.DataFrame(shifts).T.sort_values("TV", ascending=False)
    table = out.table(NAME, pd.concat(rows, names=["kernel", "population"]))
    out.table(FACTORS, pd.concat(factors, names=["kernel", "factor"]))
    shown = table[next(iter(kernels.kernels.values())).outcomes].T.rename(index=kernels.labels)
    shown.columns = [f"{label}\n{population}" for label, population in shown.columns]
    figure = grid(shown, decimals=1, scale=100, title=f"{kernels.topic.name}: what a population chooses by each kernel",
                  note="M q in percent: the run's own population, every profile counted once, and what the subjects claimed")
    return table, [out.figure(NAME, figure)]
