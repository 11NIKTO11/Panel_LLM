"""
The whole of a run against its two reference points: what the model predicted, what the same
subjects claimed, and the actual result.

Predicted and claimed come with a bootstrap interval over subjects; the actual result is a
count and has none.
"""
from typing import List, Tuple

import pandas as pd
from matplotlib.figure import Figure

from .bootstrap import HIGH, LOW, interval
from .output import Output
from .plots.comparison import source_bars
from .plots.style import REFERENCE, SECONDARY, colors
from .saved import SavedRun
from .sources import ACTUAL, CLAIMED, ESTIMATED, SOURCES, distributions, estimated, frame, ranked

NAME = "country"


def country(run: SavedRun, out: Output, draws: int = 1000, seed: int = 0, level: float = 0.95) -> Tuple[pd.DataFrame, List[Figure]]:
    """
    The run's distribution over all its subjects beside the claimed and the actual one.

    Args:
        run: the saved run.
        out: takes the table and the figure, both named `country`.
        draws, seed, level: of the bootstrap intervals.

    Returns:
        The table - one row per topic column, a column per source, `<source>_low` and
        `<source>_high` for predicted and claimed - and its figure.
    """
    topic = run.topic
    found = distributions(run, frame(run))
    edges = interval(
        frame(run),
        lambda rows: pd.concat(estimated(run, rows)),
        draws, seed, level,
    )

    table = pd.DataFrame(index=topic.columns)
    for source in SOURCES:
        if source in found:
            table[source] = found[source]
        if source in found and source in ESTIMATED:
            table[f"{source}_low"], table[f"{source}_high"] = edges.loc[source, LOW], edges.loc[source, HIGH]
    out.table(NAME, table)

    drawn = [source for source in SOURCES if source in found]
    figure = source_bars(
        table, drawn, ranked(run), {**colors(SOURCES), CLAIMED: SECONDARY, ACTUAL: REFERENCE}, head=topic.turnout or (),
        row_colors=run.style.colors or None,
        labels=run.labels, names={source: source.capitalize() for source in drawn},
        title=f"{run.spec.model} on {topic.name}",
        note=f"{len(run.subjects)} subjects of '{run.dataset.name}'; whiskers are {level:.0%} bootstrap intervals over subjects",
    )
    return table, [out.figure(NAME, figure)]
