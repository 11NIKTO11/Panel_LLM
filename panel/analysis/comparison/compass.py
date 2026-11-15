"""
Each run's map of the options beside the experts' positions, the runs side by side.

A run's numbers are those of its own `compass` analysis.
"""
from typing import List, Tuple

import pandas as pd
from matplotlib.figure import Figure

from ..compass import FIT, R, R_P, compute as single, placed
from ..output import Output
from ..plots.maps import maps
from ..saved import NotApplicable
from .loaded import RunSet

NAME = "runs_compass"


def compass(runs: RunSet, out: Output) -> Tuple[pd.DataFrame, List[Figure]]:
    """
    How well each run's map of the options agrees with the experts' positions.

    Args:
        runs: the runs to compare.
        out: takes the table and the figure, both named `runs_compass`.

    Returns:
        The table - one row per run, the columns of the single-run `compass_fit` - and its
        figure, a map per run.

    Raises:
        NotApplicable: if the topic has no expert positions.
    """
    first = next(iter(runs.runs.values()))
    positions = first.positions
    if positions is None:
        raise NotApplicable(f"topic '{runs.topic.name}' has no expert positions")
    fits, panels = {}, {}
    for label, run in runs.runs.items():
        found = single(run)
        fits[label] = found[FIT].iloc[0]
        panels[f"{label}: r = {fits[label][R]:.2f} (p = {fits[label][R_P]:.3f})"] = placed(found)
    table = pd.DataFrame(fits).T.rename_axis("run")
    out.table(NAME, table)

    x, y = positions.axes
    figure = maps(
        panels, positions.plane, positions.scale, positions.labels[x], positions.labels[y], ("the run's map,\nlaid onto the plane", "experts"),
        runs.style.colors, runs.labels,
        title=f"{runs.topic.name}: each run's map of the options",
        note="MDS of the options' correlation distances over subjects, shifted, turned and rescaled to fit the experts' positions;\nr is the Procrustes correlation, p the share of relabellings that fit at least as well",
    )
    return table, [out.figure(NAME, figure)]
