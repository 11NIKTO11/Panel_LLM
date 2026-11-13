"""
Every run area by area: how far each run's prediction, and what the subjects claimed, is from
the actual result in each area.

A run's numbers are those of its own `area` analysis, intervals included.
"""
from typing import List, Tuple

import pandas as pd
from matplotlib.figure import Figure

from ..area import PERCENT, SUBJECTS, area, slug
from ..distance import TURNOUT, shown
from ..metrics import CORRELATION
from ..output import Output
from ..plots.curves import line_panels
from ..plots.matrices import grid
from ..plots.style import EFFECTS, ERRORS, SECONDARY, colors
from ..saved import NotApplicable
from ..sources import ACTUAL, CLAIMED, PREDICTED
from .loaded import RunSet

NAME = "runs_areas"
AREA, SOURCE = "area", "source"


def areas(runs: RunSet, out: Output) -> Tuple[pd.DataFrame, List[Figure]]:
    """
    Every metric of the `area` analysis from the actual result per area, for each run and for the claimed.

    Args:
        runs: the runs to compare.
        out: takes the table, named `runs_areas`, a heatmap per metric, named
            `runs_areas_<metric>`, and `runs_areas_lines`: every metric as a line per
            source across the areas, the source's value for the whole dashed.

    Returns:
        The table - indexed by area and source, the whole first, then the areas in the
        first run's order: `subjects`, and per metric its value with `_low` and `_high` -
        and the figures: areas by sources, each cell its value with its interval underneath.

    Raises:
        NotApplicable: if a run has no area column or the topic no reference.
    """
    quiet = Output(out.store, out.stem, save=False)
    own = {label: area(run, quiet)[0] for label, run in runs.runs.items()}
    stands = {label: next(name for name, other in runs.runs.items() if other is run) for label, run in runs.claimed.items()}
    first = next(iter(own.values()))
    held = {column.split(": ")[0] for column in first.columns if column.endswith(f": {PREDICTED} vs {ACTUAL}")}
    metrics = [metric for metric in shown(True) if metric in held]

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
        scale, decimals = (100, 1) if metric in PERCENT else (1, 3)
        low, high = ((wide(f"{metric}{end}") * scale).round(decimals).astype(str) for end in ("_low", "_high"))
        return "(" + low + " - " + high + ")"

    key = next(iter(runs.runs.values())).dataset.area
    figures = [
        out.figure(f"{NAME}_{slug(metric)}", grid(
            wide(metric), below=edges(metric), decimals=1 if metric in PERCENT else 3, scale=100 if metric in PERCENT else 1,
            cmap=EFFECTS if metric == CORRELATION else ERRORS, signed=metric == TURNOUT,
            title=f"{runs.topic.name}: {metric} from the actual result by {key}",
            note=("percentage points, source minus actual" if metric == TURNOUT else "percentage points, over the options" if metric in PERCENT else "over the options")
            + "; in brackets the 95% bootstrap interval over the area's subjects",
        ))
        for metric in metrics
    ]
    total = next(iter(runs.runs.values())).total
    lines = line_panels(
        {metric: wide(metric).drop(index=names[total]).rename(index={shown: plain for plain, shown in names.items()}) for metric in metrics},
        levels={metric: wide(metric).loc[names[total]] for metric in metrics},
        colors={**dict.fromkeys(stands, SECONDARY), **colors(list(own))}, percent=[metric for metric in metrics if metric in PERCENT], zero=[TURNOUT],
        title=f"{runs.topic.name}: distance from the actual result across the {key}s",
        note=f"a line per source over the areas in their declared order; dashed, the source's value for {total}",
    )
    return table, [*figures, out.figure(f"{NAME}_lines", lines)]
