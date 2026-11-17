"""
Shapley effects: the variance over the profiles that the factors explain together, split
fairly among them (`stats.shapley`), for the answers and for the profiles' scores on the
first SVD axes.

ε² (`importance`) credits each factor with everything it explains alone, so related factors
of a sampled kernel - region and district, income and living standard - are counted twice;
Shapley effects add up to what all explain together. Columns a dataset declares `together`
are one player. On a designed kernel the factors are independent and the two nearly agree.

Intervals by the kernel's replicates (`replicates.scheme_of`): half-samples of a sampled kernel's
profiles, resampled repeats of a designed one, none without either. Both are re-centred on the
full estimate: an out-of-sample R² is lower on half the profiles, and lower again on cell means
of repeats drawn with replacement, which are noisier than the means of the repeats themselves.
"""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.figure import Figure

from ..estimate import high, low
from ..output import Output
from ..plots.intervals import interval_panels
from ..stats.replicates import interval
from ..stats.shapley import shapley_values
from ..stats.spectral import centred_svd
from .factors import players as factor_players
from .matrix import KernelSet
from .replicates import replica_answers, scheme_of

NAME = "kernel_shapley"
EFFECT, SHARE, EXPLAINED = "effect", "share of explained", "explained by all"
LOW, HIGH = low(EFFECT), high(EFFECT)
AXES = 2


def _effects(players: Dict[str, np.ndarray], values: np.ndarray) -> pd.Series:
    effects, full = shapley_values(players, values)
    return pd.concat([effects, pd.Series({EXPLAINED: full})])


def shapley(kernels: KernelSet, out: Output, draws: int = 20, seed: int = 0, jobs: int = -1) -> Tuple[pd.DataFrame, List[Figure]]:
    """`draws` replicates make the intervals, 0 for none; `jobs` are joblib workers, -1 for all cores."""
    rows, panels = {}, {}
    for label, kernel in kernels.kernels.items():
        players, values = factor_players(kernel), kernel.matrix.to_numpy().T
        scheme = scheme_of(kernel) if draws else None
        tasks = [({name: block[rows] for name, block in players.items()}, answers) for rows, answers in replica_answers(kernel, draws, seed)] if draws else []
        found = Parallel(n_jobs=jobs)(delayed(_effects)(*task) for task in [(players, values), *tasks])
        full, replicates = found[0], pd.DataFrame(found[1:])
        _, _, right = centred_svd(kernel.matrix.to_numpy())
        table = pd.DataFrame({EFFECT: full.drop(EXPLAINED)})
        bounds = {name: interval(replicates[name], full[name], recentre=True) if len(replicates) else (np.nan, np.nan) for name in table.index}
        table[LOW], table[HIGH] = [bounds[name][0] for name in table.index], [bounds[name][1] for name in table.index]
        table[SHARE] = table[EFFECT] / full[EXPLAINED]
        table[EXPLAINED] = full[EXPLAINED]
        for k in range(min(AXES, len(right))):
            table[f"axis {k + 1}"] = shapley_values(players, right[k])[0]
        table["intervals"] = f"{scheme}, {draws}" if scheme else "none"
        rows[label] = table.sort_values(EFFECT, ascending=False)
        panels[label] = rows[label][[EFFECT, LOW, HIGH]].rename(columns=lambda column: column.replace(EFFECT, label))
    result = out.table(NAME, pd.concat(rows, names=["kernel", "player"]))
    together = pd.concat(panels.values(), axis=1)
    figure = interval_panels(
        together, list(kernels.kernels), percent=list(kernels.kernels), unit="%", bars=True,
        title=f"{kernels.topic.name}: what each factor explains, shared fairly",
        note=f"Shapley effects on the leave-one-out ridge R² of the answers; 95 % intervals from {draws} replicates where a kernel has them",
    )
    return result, [out.figure(NAME, figure)]
