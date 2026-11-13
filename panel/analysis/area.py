"""
The run area by area: how far predicted, claimed and actual are from each other in each area
the topic has a result for, and over the whole.

The area is the dataset's `area` column, whose values name rows of the topic's reference. An
area is only as good as its subjects are many, so their count comes with every row. Areas
come in the order the dataset declares for the column (`levels`, e.g. by population), else
the one with most subjects first.
"""
from typing import List, Sequence, Tuple, Union

import pandas as pd
from matplotlib.figure import Figure

from .bootstrap import HIGH, LOW, interval
from .distance import TURNOUT, shown
from .metrics import MAE, RMSE, distances
from .output import Output
from .plots.intervals import interval_panels
from .saved import NotApplicable, SavedRun
from .sources import ACTUAL, CLAIMED, PREDICTED, actual, estimated, frame

NAME = "area"
SHARES = "area_shares"
PAIRS = ((PREDICTED, ACTUAL), (PREDICTED, CLAIMED), (CLAIMED, ACTUAL))
SUBJECTS = "subjects"
PERCENT = (MAE, RMSE, TURNOUT)


def slug(metric: str) -> str:
    """A metric's name as part of a file name."""
    return metric.lower().replace(" ", "_")


def ordered(run: SavedRun) -> List[str]:
    """
    The areas the run's subjects are in: in the dataset's declared order, else most subjects first.

    Raises:
        NotApplicable: if the dataset names no area column.
        ValueError: on an area the declared order does not hold.
    """
    key = run.dataset.area
    if not key:
        raise NotApplicable(f"'{run.stem}': the dataset names no area column")
    counts = run.subjects[key].value_counts()
    declared = run.dataset.levels.get(key)
    if not (declared and declared.order):
        return list(counts.index)
    unknown = sorted(set(counts.index) - set(declared.order))
    if unknown:
        raise ValueError(f"'{key}' has values its declared order does not hold: {unknown}")
    return [name for name in declared.order if name in counts.index]


def area(
    run: SavedRun, out: Output, metrics: Union[Sequence[str], None] = None, draws: int = 300, seed: int = 0, level: float = 0.95,
) -> Tuple[pd.DataFrame, List[Figure]]:
    """
    The distance between each pair of sources per area, the whole first, then the areas as `ordered` gives them.

    Args:
        run: the saved run.
        out: takes the table, named `area`, a figure per metric, named `area_<metric>`, and
            a second table `area_shares` holding every source's distribution per area.
        metrics: of `distance.shown`, one figure each; all of them by default.
        draws, seed, level: of the bootstrap intervals, resampling subjects within an area.

    Returns:
        The table - one row per area, `subjects`, and per metric and pair a column
        `<metric>: <source> vs <reference>` with `_low` and `_high` - and the figures.

    Raises:
        NotApplicable: if the dataset names no area column or the topic has no reference.
    """
    key, options, turnout = run.dataset.area, run.options, run.topic.turnout
    if not key or run.reference is None:
        raise NotApplicable(f"'{run.stem}': by area needs the dataset's area column and the topic's reference")
    metrics = list(metrics or shown(bool(turnout)))
    over = [metric for metric in metrics if metric != TURNOUT]
    rows = frame(run).join(run.subjects[[key]])
    parts = {run.total: rows, **{name: rows[rows[key] == name] for name in ordered(run)}}

    records, shares = {}, {}
    for name, part in parts.items():
        counted = actual(run, name)

        def measure(drawn: pd.DataFrame) -> pd.Series:
            found, values = {**estimated(run, drawn), **counted}, {}
            for source, reference in PAIRS:
                if source in found and reference in found:
                    for metric, value in distances(found[reference][options].to_numpy(), found[source][options].to_numpy(), over).items():
                        values[f"{metric}: {source} vs {reference}"] = value
                    if TURNOUT in metrics:
                        values[f"{TURNOUT}: {source} vs {reference}"] = found[source][turnout[0]] - found[reference][turnout[0]]
            return pd.Series(values, dtype=float)

        edges = interval(part, measure, draws, seed, level)
        record = {SUBJECTS: part.index.nunique()}
        for column, value in measure(part).items():
            record[column], record[f"{column}_low"], record[f"{column}_high"] = value, edges.loc[column, LOW], edges.loc[column, HIGH]
        records[name] = record
        for source, found in {**estimated(run, part), **counted}.items():
            shares[(name, source)] = found
    table = pd.DataFrame(records).T.astype({SUBJECTS: int})
    table.index.name = key
    out.table(SHARES, pd.DataFrame(shares).T.rename_axis([key, "source"]))
    out.table(NAME, table)

    labelled = table.rename(index=lambda name: f"{name}  (n={table.loc[name, SUBJECTS]})")
    figures = []
    for metric in metrics:
        columns = [column for column in table.columns if column.startswith(f"{metric}: ") and not column.endswith(("_low", "_high"))]
        figure = interval_panels(
            labelled, columns, percent=columns if metric in PERCENT else (), zero=columns if metric == TURNOUT else (),
            names={column: column.split(": ", 1)[1] for column in columns}, same_scale=True,
            title=f"{run.spec.model} on {run.topic.name}: {metric} by {key}",
            note=("source minus reference" if metric == TURNOUT else f"over the {len(options)} options") + f"; whiskers are {level:.0%} bootstrap intervals over the area's subjects",
        )
        figures.append(out.figure(f"{NAME}_{slug(metric)}", figure))
    return table, figures
