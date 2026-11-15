"""
How far the sources are from each other over the whole run: predicted from actual, predicted
from claimed, claimed from actual.

Each pair is a source held against a reference. The intervals come from one resampling of
subjects for all pairs, so predicted and claimed move together, as they do in truth: both are
about the same subjects.
"""
from typing import Dict, List, Tuple

import pandas as pd
from matplotlib.figure import Figure

from .estimate import high, low
from .output import Output, save
from .plots.intervals import interval_panels
from .saved import NotApplicable, SavedRun
from .sources import TURNOUT, actual, estimated, frame, pair_distances, shown, versus
from .stats.bootstrap import HIGH, LOW, intervals
from .stats.metrics import MAE, METRICS, RMSE

NAME = "distance"


def compute(run: SavedRun, draws: int = 1000, seed: int = 0, level: float = 0.95) -> Dict[str, pd.DataFrame]:
    """
    Every metric of `stats.metrics` for each pair of sources the run has, with bootstrap intervals.

    Args:
        run: the saved run.
        draws, seed, level: of the bootstrap intervals.

    Returns:
        `distance` - one row per pair `<source> vs <reference>`, a column per metric over the
        options, `Turnout difference` (source minus reference) on a topic with turnout, and
        `<metric>_low`, `<metric>_high`.

    Raises:
        NotApplicable: if the run has neither claims nor a reference, so no pair.
    """
    counted = actual(run)
    measures = [*METRICS, *([TURNOUT] if run.topic.turnout else [])]

    def measure(rows: pd.DataFrame) -> pd.Series:
        found = pair_distances(run, {**estimated(run, rows), **counted}, measures)
        return pd.Series({(versus(source, reference), metric): value for (source, reference, metric), value in found.items()}, dtype=float)

    measured = measure(frame(run))
    if measured.empty:
        raise NotApplicable(f"'{run.stem}' has neither claims nor a reference; there is nothing to measure a distance to")
    edges = intervals(frame(run), measure, draws, seed, level)
    metrics = shown(bool(run.topic.turnout))
    pairs = list(dict.fromkeys(pair for pair, _ in measured.index))
    table = measured.unstack().loc[pairs, metrics]
    for metric in metrics:
        table[low(metric)] = edges[LOW].unstack().loc[pairs, metric]
        table[high(metric)] = edges[HIGH].unstack().loc[pairs, metric]
    return {NAME: table}


def draw(run: SavedRun, tables: Dict[str, pd.DataFrame], level: float = 0.95) -> Dict[str, Figure]:
    """A panel per metric, a bar per pair with its interval, as `distance`."""
    turnout = run.topic.turnout
    metrics = shown(bool(turnout))
    figure = interval_panels(
        tables[NAME], metrics, per_row=3, percent=[metric for metric in (MAE, RMSE, TURNOUT) if metric in metrics], zero=[TURNOUT] if turnout else [], bars=True,
        title=f"{run.title}: distances between sources",
        note=f"{len(run.subjects)} subjects of '{run.dataset.name}', over the {len(run.options)} options; whiskers are {level:.0%} bootstrap intervals over subjects",
    )
    return {NAME: figure}


def distance(run: SavedRun, out: Output, draws: int = 1000, seed: int = 0, level: float = 0.95) -> Tuple[pd.DataFrame, List[Figure]]:
    """
    Every metric for each pair of sources the run has, with bootstrap intervals.

    Args:
        run: the saved run.
        out: takes the table and the figure, both named `distance`.
        draws, seed, level: of the bootstrap intervals.

    Returns:
        The table of `compute` and its figure.

    Raises:
        NotApplicable: if the run has neither claims nor a reference, so no pair.
    """
    tables = compute(run, draws, seed, level)
    return save(out, tables, draw(run, tables, level))
