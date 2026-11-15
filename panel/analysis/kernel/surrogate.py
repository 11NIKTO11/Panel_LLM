"""
The kernel approximated without interactions, `softmax(b + Σ per-factor effects)`
(`stats.surrogate.LogitAdditive`), and what that approximation says beyond the observed profiles.

- additivity: the out-of-sample R² of the surrogate's answers (5 folds);
- the largest TV between two profiles: observed (`scales`' `δ`), between the surrogate's answers
  for the observed profiles, and over every combination of levels, searched;
- each outcome's range: observed, and over the full product, searched and bounded analytically;
- extrapolation: the population with the own population's factors made independent (each
  drawn from its own distribution) pushed through the kernel, the combinations no subject had
  answered by the surrogate - how much the factors' correlation in the population matters.

A designed kernel observes the whole product: the searches return what was observed, and only
additivity and the bounds say anything new.
"""
from typing import List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.model_selection import KFold

from ..estimate import bounds
from ..output import Output
from ..plots.intervals import interval_panels
from ..stats.distributions import pairwise_tv, tv
from ..stats.surrogate import LogitAdditive
from .factors import level_designs
from .matrix import Kernel, KernelSet, push

NAME, RANGES = "kernel_surrogate", "kernel_surrogate_ranges"
OBSERVED, PRODUCT = "observed", "product"
STARTS, ROW_STARTS, DRAWS = 100, 20, 20000


def ends(name: str) -> Tuple[str, str]:
    return bounds(name)[1:]


def additive_r2(kernel: Kernel, designs: dict, folds: int = 5, seed: int = 0) -> float:
    """The surrogate's out-of-sample R², pooled over the outcomes."""
    values = kernel.matrix.to_numpy().T
    predicted = np.zeros_like(values)
    for train, test in KFold(folds, shuffle=True, random_state=seed).split(values):
        fitted = LogitAdditive(designs, kernel.factors.iloc[train], values[train])
        predicted[test] = fitted.predict(kernel.factors.iloc[test])
    return float(1 - ((values - predicted) ** 2).sum() / ((values - values.mean(axis=0)) ** 2).sum())


def independent(kernel: Kernel, surrogate: LogitAdditive, draws: int, seed: int) -> Tuple[pd.Series, float]:
    """`M q` for the own population with its factors made independent, and the share of that population the kernel observed."""
    random = np.random.default_rng(seed)
    weights = kernel.weights.reindex(kernel.factors.index).to_numpy()
    subjects = kernel.factors.loc[kernel.factors.index.repeat(weights)]
    drawn = pd.DataFrame({column: random.choice(subjects[column].to_numpy(), draws) for column in kernel.factors.columns})
    population = drawn.value_counts()
    known = pd.Series(kernel.factors.index, index=pd.MultiIndex.from_frame(kernel.factors))
    found = population.index.isin(known.index)
    observed = population[found]
    answer = pd.Series(0.0, index=kernel.outcomes)
    if len(observed):
        answer = push(kernel, pd.Series(observed.to_numpy(), index=known.loc[observed.index].to_numpy())) * observed.sum()
    unseen = population[~found]
    if len(unseen):
        answer = answer + pd.Series(surrogate.predict(unseen.index.to_frame(index=False)).T @ unseen.to_numpy(), index=kernel.outcomes)
    return answer / population.sum(), float(observed.sum() / population.sum())


def surrogate(kernels: KernelSet, out: Output, factors: Union[Sequence[str], None] = None, seed: int = 0) -> Tuple[pd.DataFrame, List[Figure]]:
    """
    Per kernel: additivity, the largest TV observed and over the full product, the pair reaching
    it, and the independent-factors population; per outcome its range observed and over the product.

    Args:
        kernels: the kernels.
        out: takes the tables and the figure.
        factors: the surrogate's factors, for every kernel; all of each kernel's by default.
        seed: of the folds, the search's starts and the independent population.

    Returns:
        The table - a row per kernel - with the outcomes' ranges beside it
        (`kernel_surrogate_ranges`), and the figure: each outcome's range, observed and over the
        full product, per kernel.
    """
    random = np.random.default_rng(seed)
    rows, ranges = {}, {}
    for label, kernel in kernels.kernels.items():
        designs = level_designs(kernel, factors)
        values = kernel.matrix.to_numpy().T
        fitted = LogitAdditive(designs, kernel.factors, values)
        predicted = fitted.predict(kernel.factors)
        observed_codes = fitted.codes(kernel.factors)
        _, (observed, first, second) = pairwise_tv(values)
        sizes = [len(design) for design in designs.values()]
        starts = [(observed_codes[first], observed_codes[second])] + [tuple(np.array([random.integers(0, size) for size in sizes]) for _ in range(2)) for _ in range(STARTS)]
        found = fitted.largest_tv(starts)
        reached = np.array([value for value, _ in found])
        best = found[int(reached.argmax())][1]
        pair = [dict(zip(designs, fitted.levels(codes))) for codes in best]
        differing = {column: f"{pair[0][column]} | {pair[1][column]}" for column in designs if pair[0][column] != pair[1][column]}
        low, high = fitted.bounds()
        row_starts = [(codes,) for codes in observed_codes[random.choice(len(observed_codes), min(ROW_STARTS, len(observed_codes)), replace=False)]]
        searched_high = np.array([max(value for value, _ in fitted.extreme(k, 1.0, row_starts)) for k in range(len(kernel.outcomes))])
        searched_low = np.array([-max(value for value, _ in fitted.extreme(k, -1.0, row_starts)) for k in range(len(kernel.outcomes))])
        mixed, seen = independent(kernel, fitted, DRAWS, seed)
        own = push(kernel, kernel.weights)
        rows[label] = {
            "additive R² (out of sample)": additive_r2(kernel, designs, seed=seed), "δ observed": observed,
            "δ surrogate, observed profiles": pairwise_tv(predicted)[1][0], "δ surrogate, full product": float(reached.max()),
            "starts within 0.01 of it": float((reached >= reached.max() - 0.01).mean()),
            "full product pair, differing": "; ".join(f"{column}: {value}" for column, value in differing.items()),
            "independent factors: TV from own": tv(mixed.to_numpy(), own.to_numpy()), "independent factors: share observed": seen,
        }
        ranges[label] = pd.DataFrame({
            "mean": values.mean(axis=0), ends(OBSERVED)[0]: values.min(axis=0), ends(OBSERVED)[1]: values.max(axis=0),
            ends(PRODUCT)[0]: searched_low, ends(PRODUCT)[1]: searched_high, "bound low": low, "bound high": high,
        }, index=kernel.outcomes)
    table = out.table(NAME, pd.DataFrame(rows).T.rename_axis("kernel"))
    out.table(RANGES, pd.concat(ranges, names=["kernel", "outcome"]))
    shown = {}
    for label, frame in ranges.items():
        for kind in (OBSERVED, PRODUCT):
            shown[f"{label} {kind}"] = frame["mean"]
            shown[ends(f"{label} {kind}")[0]], shown[ends(f"{label} {kind}")[1]] = frame[ends(kind)[0]], frame[ends(kind)[1]]
    panels = pd.DataFrame(shown).rename(index=kernels.labels)
    figure = interval_panels(
        panels, [f"{label} {kind}" for label in ranges for kind in (OBSERVED, PRODUCT)], percent=list(panels.columns), unit="%", same_scale=True,
        per_row=2, title=f"{kernels.topic.name}: each outcome's range, observed and over every combination",
        note="dot: the outcome's mean over the observed profiles; bars: its lowest and highest probability over them, and over the full product by the additive surrogate",
    )
    return table, [out.figure(NAME, figure)]
