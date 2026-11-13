"""
How much each run tells the subjects of one area apart: the standard deviation of every
option's share over the area's subjects, the runs one under the other.

A subject is one row, repeats averaged. A run that asks one subject per area has no spread
within an area and is left out.
"""
from typing import List, Tuple

import pandas as pd
from matplotlib.figure import Figure

from ..aggregate import subject_answers
from ..area import ordered
from ..output import Output
from ..plots.matrices import grids
from ..plots.style import EFFECTS
from ..saved import NotApplicable
from ..sources import ranked
from .loaded import RunSet

NAME = "runs_spread"
RUN, AREA = "run", "area"


def spreads(runs: RunSet, out: Output) -> Tuple[pd.DataFrame, List[Figure]]:
    """
    The standard deviation of each option's share within each area, per run.

    Args:
        runs: the runs to compare.
        out: takes the table and the figure, both named `runs_spread`.

    Returns:
        The table - indexed by run and area, a column per option - and the figure: per run
        a panel of options by areas, in percentage points, on one colour range.

    Raises:
        NotApplicable: if no run has several subjects in an area.
    """
    found = {}
    for label, run in runs.runs.items():
        answers = subject_answers(run.normalized, run.topic)[run.options]
        within = answers.groupby(run.subjects.loc[answers.index, run.dataset.area]).std().reindex(ordered(run))
        if within.notna().any().any():
            found[label] = within
    if not found:
        raise NotApplicable(f"no run of '{runs.topic.name}' has several subjects in an area")
    table = out.table(NAME, pd.concat(found, names=[RUN, AREA]))
    first = next(run for label, run in runs.runs.items() if label in found)
    figure = grids(
        {label: within[ranked(first)].T.rename(index=first.labels) for label, within in found.items()}, decimals=1, scale=100, cmap=EFFECTS,
        title=f"{runs.topic.name}: how far apart a run puts the subjects of one {first.dataset.area}",
        note="standard deviation of each option's share over the area's subjects, percentage points",
    )
    return table, [out.figure(NAME, figure)]
