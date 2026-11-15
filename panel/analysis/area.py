"""
The run area by area: how far predicted, claimed and actual are from each other in each area
the topic has a result for, and over the whole.

The area is the dataset's `area` column, whose values name rows of the topic's reference. An
area is only as good as its subjects are many, so their count comes with every row. Areas
come in the order the dataset declares for the column (`levels`, e.g. by population), else
the one with most subjects first.
"""
from typing import Dict, List, Sequence, Tuple, Union

import pandas as pd
from matplotlib.figure import Figure

from .estimate import high, low
from .levels import members
from .output import Output, save
from .plots.intervals import interval_panels
from .saved import NotApplicable, SavedRun
from .sources import PAIRS, TURNOUT, actual, estimated, frame, pair_distances, shown, versus
from .stats.bootstrap import HIGH, LOW, intervals
from .stats.metrics import MAE, RMSE

NAME = "area"
SHARES = "area_shares"
SUBJECTS = "subjects"
PERCENT = (MAE, RMSE, TURNOUT)


def slug(metric: str) -> str:
    """A metric's name as part of a file name."""
    return metric.lower().replace(" ", "_")


def column(metric: str, source: str, reference: str) -> str:
    """The column of the `area` table that holds one metric of one pair."""
    return f"{metric}: {versus(source, reference)}"


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
    return list(members(run.subjects[key], run.dataset.levels.get(key)))


def compute(
    run: SavedRun, metrics: Union[Sequence[str], None] = None, draws: int = 300, seed: int = 0, level: float = 0.95,
) -> Dict[str, pd.DataFrame]:
    """
    The distance between each pair of sources per area, the whole first, then the areas as `ordered` gives them.

    Args:
        run: the saved run.
        metrics: of `sources.shown`; all of them by default.
        draws, seed, level: of the bootstrap intervals, resampling subjects within an area.

    Returns:
        `area` - one row per area, `subjects`, and per metric and pair a column
        `<metric>: <source> vs <reference>` (`column`) with `_low` and `_high` - and
        `area_shares`, every source's distribution per area.

    Raises:
        NotApplicable: if the dataset names no area column or the topic has no reference.
    """
    key = run.dataset.area
    if not key or run.reference is None:
        raise NotApplicable(f"'{run.stem}': by area needs the dataset's area column and the topic's reference")
    metrics = list(metrics or shown(bool(run.topic.turnout)))
    rows = frame(run).join(run.subjects[[key]])
    parts = {run.total: rows, **{name: rows[rows[key] == name] for name in ordered(run)}}

    records, shares = {}, {}
    for name, part in parts.items():
        counted = actual(run, name)

        def measure(drawn: pd.DataFrame) -> pd.Series:
            found = pair_distances(run, {**estimated(run, drawn), **counted}, metrics)
            return pd.Series({column(metric, source, reference): value for (source, reference, metric), value in found.items()}, dtype=float)

        edges = intervals(part, measure, draws, seed, level)
        record = {SUBJECTS: part.index.nunique()}
        for held, value in measure(part).items():
            record[held], record[low(held)], record[high(held)] = value, edges.loc[held, LOW], edges.loc[held, HIGH]
        records[name] = record
        for source, found in {**estimated(run, part), **counted}.items():
            shares[(name, source)] = found
    table = pd.DataFrame(records).T.astype({SUBJECTS: int})
    table.index.name = key
    return {NAME: table, SHARES: pd.DataFrame(shares).T.rename_axis([key, "source"])}


def draw(run: SavedRun, tables: Dict[str, pd.DataFrame], metrics: Union[Sequence[str], None] = None, level: float = 0.95) -> Dict[str, Figure]:
    """A figure per metric, named `area_<metric>`: a panel per pair, a row per area with its interval."""
    table, key = tables[NAME], run.dataset.area
    labelled = table.rename(index=lambda name: f"{name}  (n={table.loc[name, SUBJECTS]})")
    figures = {}
    for metric in metrics or shown(bool(run.topic.turnout)):
        pairs = {column(metric, source, reference): versus(source, reference) for source, reference in PAIRS}
        columns = [held for held in pairs if held in table.columns]
        figures[f"{NAME}_{slug(metric)}"] = interval_panels(
            labelled, columns, percent=columns if metric in PERCENT else (), zero=columns if metric == TURNOUT else (),
            names={held: pairs[held] for held in columns}, same_scale=True,
            title=f"{run.title}: {metric} by {key}",
            note=("source minus reference" if metric == TURNOUT else f"over the {len(run.options)} options") + f"; whiskers are {level:.0%} bootstrap intervals over the area's subjects",
        )
    return figures


def area(
    run: SavedRun, out: Output, metrics: Union[Sequence[str], None] = None, draws: int = 300, seed: int = 0, level: float = 0.95,
) -> Tuple[pd.DataFrame, List[Figure]]:
    """
    The distance between each pair of sources per area, with a figure per metric.

    Args:
        run: the saved run.
        out: takes the table, named `area`, a second table `area_shares`, and a figure per
            metric, named `area_<metric>`.
        metrics: of `sources.shown`, one figure each; all of them by default.
        draws, seed, level: of the bootstrap intervals, resampling subjects within an area.

    Returns:
        The `area` table of `compute` and the figures.

    Raises:
        NotApplicable: if the dataset names no area column or the topic has no reference.
    """
    tables = compute(run, metrics, draws, seed, level)
    return save(out, tables, draw(run, tables, metrics, level))
