"""
The run area by area: how far predicted, claimed and actual are from each other in each area
the topic has a result for, and over the whole.

The area is the dataset's `area` column, whose values name rows of the topic's reference. An
area is only as good as its subjects are many, so their count comes with every row.
"""
from typing import List, Sequence, Tuple

import pandas as pd
from matplotlib.figure import Figure

from .bootstrap import HIGH, LOW, interval
from .metrics import MAE, RMSE, distances
from .output import Output
from .plots.intervals import interval_panels
from .saved import NotApplicable, SavedRun
from .sources import ACTUAL, CLAIMED, PREDICTED, actual, estimated, frame

NAME = "area"
SHARES = "area_shares"
PAIRS = ((PREDICTED, ACTUAL), (PREDICTED, CLAIMED), (CLAIMED, ACTUAL))
SUBJECTS = "subjects"


def area(
    run: SavedRun, out: Output, metrics: Sequence[str] = (MAE, RMSE), draws: int = 300, seed: int = 0, level: float = 0.95,
) -> Tuple[pd.DataFrame, List[Figure]]:
    """
    The distance between each pair of sources per area, the whole first, then areas by subject count.

    Args:
        run: the saved run.
        out: takes the table, named `area`, a figure per metric, named `area_<metric>`, and
            a second table `area_shares` holding every source's distribution per area.
        metrics: of `metrics`, one figure each.
        draws, seed, level: of the bootstrap intervals, resampling subjects within an area.

    Returns:
        The table - one row per area, `subjects`, and per metric and pair a column
        `<metric>: <source> vs <reference>` with `_low` and `_high` - and the figures.

    Raises:
        NotApplicable: if the dataset names no area column or the topic has no reference.
    """
    key, options = run.dataset.area, run.options
    if not key or run.reference is None:
        raise NotApplicable(f"'{run.stem}': by area needs the dataset's area column and the topic's reference")
    rows = frame(run).join(run.subjects[[key]])
    counts = run.subjects[key].value_counts()
    parts = {run.total: rows, **{name: rows[rows[key] == name] for name in counts.index}}

    records, shares = {}, {}
    for name, part in parts.items():
        counted = {source: found[options].to_numpy() for source, found in actual(run, name).items()}

        def measure(drawn: pd.DataFrame) -> pd.Series:
            found = {**{source: shares[options].to_numpy() for source, shares in estimated(run, drawn).items()}, **counted}
            return pd.Series({
                f"{metric}: {source} vs {reference}": value
                for source, reference in PAIRS if source in found and reference in found
                for metric, value in distances(found[reference], found[source], metrics).items()
            }, dtype=float)

        edges = interval(part, measure, draws, seed, level)
        record = {SUBJECTS: part.index.nunique()}
        for column, value in measure(part).items():
            record[column], record[f"{column}_low"], record[f"{column}_high"] = value, edges.loc[column, LOW], edges.loc[column, HIGH]
        records[name] = record
        for source, found in {**estimated(run, part), **actual(run, name)}.items():
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
            labelled, columns, percent=columns, names={column: column.split(": ", 1)[1] for column in columns}, same_scale=True,
            title=f"{run.spec.model} on {run.topic.name}: {metric} by {key}",
            note=f"over the {len(options)} options; whiskers are {level:.0%} bootstrap intervals over the area's subjects",
        )
        figures.append(out.figure(f"{NAME}_{metric.lower()}", figure))
    return table, figures
