"""
Minimal pairs: two profiles that differ in exactly one factor, and how far apart their answers are.

The TV of such a pair is what swapping that one value does to the answer, everything else held.
Read against the noise floor: a swap that moves the answer no more than asking again does
nothing. A designed kernel pairs every profile with every other level of every factor, so each
factor gets the whole distribution of its swaps; a sampled one has the pairs its subjects
happen to form, most of them in the factors with many values.

Values are compared as the dataset writes them: two ages are two values.
"""
from typing import List, Tuple

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..output import Output
from ..plots.distributions import strips
from ..saved import NotApplicable
from ..stats.distributions import tv
from .matrix import Kernel, KernelSet
from .noise import noise_floor

NAME, EACH = "kernel_pairs", "kernel_pairs_each"
FACTOR, FIRST, SECOND, DISTANCE = "factor", "first", "second", "TV"


def minimal_pairs(kernel: Kernel) -> pd.DataFrame:
    """
    Every pair of the kernel's profiles that differs in one factor: `factor`, the two profiles,
    the two values, and the TV of their answers.
    """
    factors = kernel.factors
    codes = np.column_stack([pd.factorize(factors[column])[0] for column in factors.columns])
    differ = sum((codes[:, k][:, None] != codes[:, k][None]).astype(np.int8) for k in range(codes.shape[1]))
    first, second = np.nonzero(np.triu(differ == 1, 1))
    which = (codes[first] != codes[second]).argmax(axis=1)
    columns = kernel.matrix.to_numpy().T
    names = np.asarray(factors.columns)[which]
    values = factors.to_numpy()
    return pd.DataFrame({
        FACTOR: names, FIRST: factors.index[first], SECOND: factors.index[second],
        "from": values[first, which], "to": values[second, which], DISTANCE: tv(columns[first], columns[second]),
    })


def pairs(kernels: KernelSet, out: Output) -> Tuple[pd.DataFrame, List[Figure]]:
    rows, each, panels, marks = {}, {}, {}, {}
    for label, kernel in kernels.kernels.items():
        found = minimal_pairs(kernel)
        if found.empty:
            continue
        floor = noise_floor(kernel).floor
        stats = found.groupby(FACTOR)[DISTANCE].agg(["count", "mean", "median", lambda values: values.quantile(0.9), "max"])
        stats.columns = ["pairs", "mean TV", "median TV", "q90 TV", "largest TV"]
        stats["mean / noise floor"] = stats["mean TV"] / floor
        rows[label] = stats.sort_values("mean TV", ascending=False)
        each[label] = found
        panels[label] = {factor: found.loc[found[FACTOR] == factor, DISTANCE].to_numpy() for factor in rows[label].index}
        if not np.isnan(floor):
            marks[label] = floor
    if not rows:
        raise NotApplicable("no kernel has two profiles that differ in one factor only")
    table = out.table(NAME, pd.concat(rows, names=["kernel", FACTOR]))
    out.table(EACH, pd.concat(each, names=["kernel", "pair"]))
    figure = strips(
        panels, marks, xlabel="TV between the answers of the two profiles",
        title=f"{kernels.topic.name}: what swapping one value does to the answer",
        note="every pair of profiles that differ in one factor only, by that factor, largest mean first; dashed: the noise floor",
    )
    return table, [out.figure(NAME, figure)]
