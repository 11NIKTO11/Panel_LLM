"""
Every run's distribution over all its subjects, side by side, with the actual result, what the
subjects claimed and the published polls.

Each run's numbers are those of its own `country` analysis, intervals included.
"""
from typing import List, Tuple

import pandas as pd
from matplotlib.figure import Figure

from ..country import NAME as COUNTRY, compute as country
from ..estimate import bounds, is_end
from ..output import Output
from ..plots.comparison import source_bars
from ..plots.style import REFERENCE, SECONDARY, colors
from ..sources import ACTUAL, CLAIMED, PREDICTED, ranked
from .loaded import RunSet

NAME = "runs_shares"


def shares(runs: RunSet, out: Output) -> Tuple[pd.DataFrame, List[Figure]]:
    """
    The table is one row per topic column, a column per source: actual, claimed, each run, each
    poll. A poll has no turnout and is not drawn.
    """
    topic = runs.topic
    own = {label: country(run)[COUNTRY] for label, run in runs.runs.items()}
    stands = runs.stands

    table = pd.DataFrame(index=topic.columns)
    first = next(iter(own.values()))
    if ACTUAL in first:
        table[ACTUAL] = first[ACTUAL]
    for label, name in stands.items():
        for mine, theirs in zip(bounds(label), bounds(CLAIMED)):
            table[mine] = own[name][theirs]
    for label, found in own.items():
        for mine, theirs in zip(bounds(label), bounds(PREDICTED)):
            table[mine] = found[theirs]
    drawn = [column for column in table.columns if not is_end(column)]
    for poll, row in runs.polls.iterrows():
        table[poll] = row
    out.table(NAME, table)

    figure = source_bars(
        table, drawn, ranked(next(iter(runs.runs.values()))), {ACTUAL: REFERENCE, **dict.fromkeys(stands, SECONDARY), **colors(list(own))}, head=topic.turnout or (),
        labels=runs.labels,
        title=f"{topic.name}: {len(runs.runs)} runs side by side",
        note="whiskers are 95% bootstrap intervals over each run's subjects",
    )
    return table, [out.figure(NAME, figure)]
