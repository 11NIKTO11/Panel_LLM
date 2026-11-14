"""
95 % intervals for the kernel's main statistics, from its replicates (`matrix.scheme_of`).

- a sampled kernel: half-samples of its profiles, the percentile interval of the replicates
  (`stats.replicates`) - what another draw of subjects from the population would give;
- a designed kernel: its repeats resampled, the replicates' spread re-centred on the estimate -
  a resampled cell mean is noisier than the mean of the repeats themselves, which biases every
  statistic of the spread between cells; what another round of the model's answers would give;
- a designed kernel without repeats, or of delivered means: no interval.

The statistics: ε² of each factor, the share of the first axis and of the first two, the
effective number of axes, the mean TV from the mean answer, and the median and 99th percentile
of the TV between two profiles (over a fixed number of random pairs, so a replicate costs the
same as the estimate). With a reference kernel of the same profiles, the replicates are paired
- the same half for both - and every difference from the reference gets its interval too.
"""
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..output import Output
from ..plots.intervals import interval_panels
from ..stats.distributions import tv
from ..stats.replicates import interval
from ..stats.spectral import EFFECTIVE_AXES, centred_svd, soft_ranks
from ..stats.variance import eps2
from .matrix import HALVES, REPEATS, Kernel, KernelSet, replicas, scheme_of

NAME, DIFFERENCES = "kernel_intervals", "kernel_intervals_differences"
ESTIMATE, LOW, HIGH = "estimate", "low", "high"
PAIRS = 20000


def statistics(kernel: Kernel, pairs: int = PAIRS, seed: int = 0) -> pd.Series:
    """The statistics one replicate is summed up by; see the module docstring."""
    values = kernel.matrix.to_numpy().T
    found = {f"ε² {column}": eps2(values, kernel.grouping(column).to_numpy())[0] for column in kernel.factors.columns}
    _, singular, _ = centred_svd(kernel.matrix.to_numpy())
    shares = singular ** 2 / (singular ** 2).sum()
    found["axis 1 share"], found["axes 1-2 share"] = shares[0], shares[:2].sum()
    found["effective number of axes"] = soft_ranks(singular)[EFFECTIVE_AXES]
    found["mean TV from the mean answer"] = float(tv(values, values.mean(axis=0)).mean())
    drawn = np.random.default_rng(seed).integers(0, len(values), (2, pairs))
    distances = tv(values[drawn[0]], values[drawn[1]])
    found["median pairwise TV"], found["q99 pairwise TV"] = float(np.median(distances)), float(np.quantile(distances, 0.99))
    return pd.Series(found, dtype=float)


def intervals(kernels: KernelSet, out: Output, draws: int = 200, seed: int = 0, reference: Union[str, None] = None) -> Tuple[pd.DataFrame, List[Figure]]:
    """
    Per kernel and statistic, the estimate and its interval; with a reference, the paired differences.

    Args:
        kernels: the kernels.
        out: takes the tables and the figure.
        draws: replicates per kernel.
        seed: of the replicates and the random pairs.
        reference: a kernel's label; the kernels of the same profiles and scheme get paired
            differences from it.

    Returns:
        The table - indexed by kernel and statistic: the estimate, low, high, and how the
        interval was made - with the differences beside it (`kernel_intervals_differences`),
        and the figure: every statistic with its interval, a panel per kernel.

    Raises:
        KeyError: on a reference that is not a kernel's label.
    """
    if reference is not None and reference not in kernels.kernels:
        raise KeyError(f"no kernel '{reference}'; there are {list(kernels.kernels)}")
    estimates, replicates, rows = {}, {}, {}
    for label, kernel in kernels.kernels.items():
        estimates[label] = statistics(kernel, seed=seed)
        replicates[label] = pd.DataFrame([statistics(replica, seed=seed) for replica in replicas(kernel, draws, seed)])
        scheme = scheme_of(kernel)
        frame = pd.DataFrame({ESTIMATE: estimates[label]})
        bounds = [interval(replicates[label][name], estimates[label][name], recentre=scheme == REPEATS) if len(replicates[label]) else (np.nan, np.nan) for name in frame.index]
        frame[LOW], frame[HIGH] = [low for low, _ in bounds], [high for _, high in bounds]
        frame["interval"] = f"{scheme}, {draws}" if scheme else "none: one answer per cell"
        rows[label] = frame
    table = out.table(NAME, pd.concat(rows, names=["kernel", "statistic"]))
    if reference is not None:
        held = kernels.kernels[reference]
        differences = {}
        for label, kernel in kernels.kernels.items():
            same = label != reference and scheme_of(kernel) == scheme_of(held) == HALVES and kernel.factors.equals(held.factors)
            if not same:
                continue
            gap = estimates[label] - estimates[reference]
            spread = replicates[label] - replicates[reference]
            frame = pd.DataFrame({ESTIMATE: gap})
            bounds = [interval(spread[name], gap[name]) for name in frame.index]
            frame[LOW], frame[HIGH] = [low for low, _ in bounds], [high for _, high in bounds]
            differences[label] = frame
        if differences:
            out.table(DIFFERENCES, pd.concat(differences, names=["kernel", "statistic"]))
    shown = {}
    for label, frame in rows.items():
        shown[label], shown[f"{label}_low"], shown[f"{label}_high"] = frame[ESTIMATE], frame[LOW], frame[HIGH]
    figure = interval_panels(
        pd.DataFrame(shown), list(rows), title=f"{kernels.topic.name}: the kernels' statistics with 95 % intervals",
        note=f"{draws} replicates: half-samples of a sampled kernel's profiles, resampled repeats of a designed one; a designed kernel with one answer per cell has none",
    )
    return table, [out.figure(NAME, figure)]
