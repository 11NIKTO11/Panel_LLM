"""
What the variation of each outcome over the profiles is made of: main effects of the factors,
their two-way interactions, the model's noise, and the rest.

- designed kernel: exact. The complete factorial splits the sum of squares into each factor's
  main effect, each pair's interaction and what is left (higher interactions), by projections
  on level and cell means (`stats.variance.factorial`).
- sampled kernel: estimated, after `llm_pushforward`. Main effects are the out-of-sample R² of a
  ridge regression on the factors' indicators; interactions what adding every pairwise product
  adds to it.

Noise is the noise variance of one answer (`noise.noise_floor`) over the answers a profile's
column averages, against the outcome's variance; on a designed kernel it is taken out of the
rest, on a sampled one out of what the factors leave. A kernel without a noise floor has none.
"""
from typing import List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..output import Output
from ..plots.stacked import stacked_bars
from ..stats.design import with_interactions
from ..stats.variance import cv_r2, factorial
from .matrix import PROFILE, Kernel, KernelSet
from .noise import noise_floor

NAME, OUTCOMES = "kernel_variance", "kernel_variance_outcomes"
MAIN, INTERACTIONS, NOISE, REST = "main effects", "interactions", "noise", "rest"
PARTS = (MAIN, INTERACTIONS, NOISE, REST)
POOLED = "pooled"


def _noise_share(kernel: Kernel, spread: np.ndarray) -> np.ndarray:
    variance = noise_floor(kernel).variance.to_numpy()
    if np.isnan(variance).all():
        return np.zeros(len(spread))
    per_column = 1.0 / kernel.calls.groupby(PROFILE).size().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nan_to_num(variance * per_column.mean() / spread)


def decompose(kernel: Kernel, factors: Union[Sequence[str], None] = None) -> pd.DataFrame:
    """
    One kernel's split, a row per outcome and one pooled, a column per part, each row summing to 1.

    Args:
        kernel: the kernel.
        factors: those the split is over; all of the kernel's by default.
    """
    columns = list(factors or kernel.factors.columns)
    values = kernel.matrix.to_numpy().T
    spread = values.var(axis=0)
    noise = _noise_share(kernel, spread)
    outcomes = kernel.outcomes + [POOLED]
    if kernel.designed and len(columns) == kernel.factors.shape[1]:
        split = factorial(values, kernel.factors[columns])
        total = split["total"]
        main = sum(split[column] for column in columns)
        pairs = sum(value for name, value in split.items() if " x " in name)
        rest = split["rest"]
        noise_ss = np.minimum(noise * total, rest)
        parts = np.column_stack([main, pairs, noise_ss, rest - noise_ss])
        parts = np.vstack([parts, parts.sum(axis=0)])
        return pd.DataFrame(parts / parts.sum(axis=1, keepdims=True), index=outcomes, columns=list(PARTS))
    blocks = list(kernel.design(columns).values())
    main, main_pooled = cv_r2(np.hstack(blocks), values)
    joint, joint_pooled = cv_r2(with_interactions(blocks), values)
    main, joint = np.append(main, main_pooled), np.append(joint, joint_pooled)
    noise = np.append(noise, (noise * spread).sum() / spread.sum())
    main = np.clip(np.nan_to_num(main), 0, 1)
    pairs = np.clip(np.nan_to_num(joint) - main, 0, 1 - main)
    noise = np.minimum(noise, 1 - main - pairs)
    return pd.DataFrame(np.column_stack([main, pairs, noise, 1 - main - pairs - noise]), index=outcomes, columns=list(PARTS))


def variance(kernels: KernelSet, out: Output, factors: Union[Sequence[str], None] = None) -> Tuple[pd.DataFrame, List[Figure]]:
    """
    Each kernel's split of the variance over its profiles.

    Args:
        kernels: the kernels.
        out: takes the tables and the figure.
        factors: the factors the split is over, for every kernel; all of each kernel's by default.

    Returns:
        The table - a row per kernel, the pooled split, the method, and the summed variance -
        with every outcome's split beside it (`kernel_variance_outcomes`), and the figure:
        a stacked bar per kernel and outcome.
    """
    split = {label: decompose(kernel, factors) for label, kernel in kernels.kernels.items()}
    rows = {}
    for label, kernel in kernels.kernels.items():
        rows[label] = {**split[label].loc[POOLED].to_dict(), "method": "exact factorial" if kernel.designed and not factors else "ridge, out of sample",
                       "summed variance": float(kernel.matrix.to_numpy().var(axis=1).sum())}
    table = out.table(NAME, pd.DataFrame(rows).T.rename_axis("kernel"))
    every = pd.concat(split, names=["kernel", "outcome"])
    out.table(OUTCOMES, every)
    bars = every.rename(index=kernels.labels, level=1)
    bars.index = [f"{label} · {outcome}" for label, outcome in bars.index]
    figure = stacked_bars(
        bars, title=f"{kernels.topic.name}: what each outcome's variation over the profiles is made of",
        note="main effects of the factors, their two-way interactions, the model's noise, and the rest; a designed kernel exactly, a sampled one by out-of-sample ridge R²",
    )
    return table, [out.figure(NAME, figure)]
