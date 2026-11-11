"""
Every run area by area: how far each run's prediction, and what the subjects claimed, is from
the actual result in each area.

A run's numbers are those of its own `area` analysis, intervals included.
"""
from typing import List, Tuple

import pandas as pd
from matplotlib.figure import Figure

from ..area import SUBJECTS, area
from ..output import Output
from ..plots.matrices import grid
from ..plots.style import ERRORS
from ..saved import NotApplicable
from ..sources import ACTUAL, CLAIMED, PREDICTED
from .loaded import RunSet

NAME = "runs_areas"
AREA, SOURCE = "area", "source"


def areas(runs: RunSet, out: Output) -> Tuple[pd.DataFrame, List[Figure]]:
    """
    MAE and RMSE from the actual result per area, for each run and for the claimed.

    Args:
        runs: the runs to compare.
        out: takes the table, named `runs_areas`, and a heatmap per metric, named
            `runs_areas_<metric>`.

    Returns:
        The table - indexed by area and source, the whole first, then areas by the first
        run's subject count: `subjects`, and per metric its value with `_low` and `_high` -
        and the figures: areas by sources, each cell its value in percentage points with
        its interval underneath.

    Raises:
        NotApplicable: if a run has no area column or the topic no reference.
    """
    quiet = Output(out.store, out.stem, save=False)
    own = {label: area(run, quiet)[0] for label, run in runs.runs.items()}
    stands = {label: next(name for name, other in runs.runs.items() if other is run) for label, run in runs.claimed.items()}
    first = next(iter(own.values()))
    metrics = [column.split(": ")[0] for column in first.columns if column.endswith(f": {PREDICTED} vs {ACTUAL}")]

    def rows(found: pd.DataFrame, source: str) -> pd.DataFrame:
        picked = {SUBJECTS: found[SUBJECTS]}
        for metric in metrics:
            for end in ("", "_low", "_high"):
                picked[f"{metric}{end}"] = found[f"{metric}: {source} vs {ACTUAL}{end}"]
        return pd.DataFrame(picked)

    parts = {**{label: rows(own[name], CLAIMED) for label, name in stands.items()}, **{label: rows(found, PREDICTED) for label, found in own.items()}}
    order = list(first.index) + sorted({name for found in own.values() for name in found.index} - set(first.index))
    table = pd.concat(parts, names=[SOURCE, AREA]).swaplevel().reindex(pd.MultiIndex.from_product([order, list(parts)], names=[AREA, SOURCE])).dropna(how="all")
    out.table(NAME, table)

    same = len({frozenset(run.subjects.index) for run in runs.runs.values()}) == 1
    names = {name: f"{name}  (n={int(first.loc[name, SUBJECTS])})" if same else name for name in order}

    def wide(column: str) -> pd.DataFrame:
        return table[column].unstack(SOURCE).reindex(index=order, columns=list(parts)).rename(index=names)

    def edges(metric: str) -> pd.DataFrame:
        low, high = ((wide(f"{metric}{end}") * 100).round(1).astype(str) for end in ("_low", "_high"))
        return "(" + low + " - " + high + ")"

    key = next(iter(runs.runs.values())).dataset.area
    figures = [
        out.figure(f"{NAME}_{metric.lower()}", grid(
            wide(metric), below=edges(metric), decimals=1, scale=100, cmap=ERRORS,
            title=f"{runs.topic.name}: {metric} from the actual result by {key}",
            note="percentage points, over the options; in brackets the 95% bootstrap interval over the area's subjects",
        ))
        for metric in metrics
    ]
    return table, figures
