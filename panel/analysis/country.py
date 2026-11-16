"""
The whole of a run against its two reference points: what the model predicted, what the same
subjects claimed, and the actual result.

Predicted and claimed come with a bootstrap interval over subjects; the actual result is a
count and has none.
"""
from typing import Dict, List, Tuple

import pandas as pd
from matplotlib.figure import Figure

from .estimate import high, low
from .output import Output, save
from .plots.comparison import source_bars
from .plots.style import REFERENCE, SECONDARY, colors
from .saved import SavedRun
from .sources import ACTUAL, CLAIMED, ESTIMATED, SOURCES, distributions, estimated, frame, ranked
from .stats.bootstrap import HIGH, LOW, intervals

NAME = "country"


def compute(run: SavedRun, draws: int = 1000, seed: int = 0, level: float = 0.95) -> Dict[str, pd.DataFrame]:
    """`country` - one row per topic column, a column per source, `<source>_low` and `<source>_high` for predicted and claimed."""
    found = distributions(run, frame(run))
    edges = intervals(frame(run), lambda rows: pd.concat(estimated(run, rows)), draws, seed, level)
    table = pd.DataFrame(index=run.topic.columns)
    for source in SOURCES:
        if source in found:
            table[source] = found[source]
        if source in found and source in ESTIMATED:
            table[low(source)], table[high(source)] = edges.loc[source, LOW], edges.loc[source, HIGH]
    return {NAME: table}


def draw(run: SavedRun, tables: Dict[str, pd.DataFrame], level: float = 0.95) -> Dict[str, Figure]:
    """The sources side by side, a group of bars per topic column, as `country`."""
    table = tables[NAME]
    drawn = [source for source in SOURCES if source in table.columns]
    figure = source_bars(
        table, drawn, ranked(run), {**colors(SOURCES), CLAIMED: SECONDARY, ACTUAL: REFERENCE}, head=run.topic.turnout or (),
        row_colors=run.style.colors or None,
        labels=run.labels, names={source: source.capitalize() for source in drawn},
        title=run.title,
        note=f"{len(run.subjects)} subjects of '{run.dataset.name}'; whiskers are {level:.0%} bootstrap intervals over subjects",
    )
    return {NAME: figure}


def country(run: SavedRun, out: Output, draws: int = 1000, seed: int = 0, level: float = 0.95) -> Tuple[pd.DataFrame, List[Figure]]:
    tables = compute(run, draws, seed, level)
    return save(out, tables, draw(run, tables, level))
