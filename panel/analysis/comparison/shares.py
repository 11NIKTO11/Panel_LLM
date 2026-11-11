"""
Every run's distribution over all its subjects, side by side, with the actual result, what the
subjects claimed and the published polls.

Each run's numbers are those of its own `country` analysis, intervals included.
"""
from typing import List, Tuple

import pandas as pd
from matplotlib.figure import Figure

from ..country import country
from ..output import Output
from ..plots.comparison import source_bars
from ..plots.style import REFERENCE, SECONDARY, colors
from ..sources import ACTUAL, CLAIMED, PREDICTED, ranked
from .loaded import RunSet

NAME = "runs_shares"


def shares(runs: RunSet, out: Output) -> Tuple[pd.DataFrame, List[Figure]]:
    """
    One column per source over the topic's columns: actual, claimed, each run, each poll.

    Args:
        runs: the runs to compare.
        out: takes the table and the figure, both named `runs_shares`.

    Returns:
        The table - one row per topic column; `<source>_low` and `<source>_high` for the
        runs and the claimed; a poll has no turnout - and its figure, polls not drawn.
    """
    topic = runs.topic
    quiet = Output(out.store, out.stem, save=False)
    own = {label: country(run, quiet)[0] for label, run in runs.runs.items()}
    stands = {label: next(name for name, other in runs.runs.items() if other is run) for label, run in runs.claimed.items()}

    table = pd.DataFrame(index=topic.columns)
    first = next(iter(own.values()))
    if ACTUAL in first:
        table[ACTUAL] = first[ACTUAL]
    for label, name in stands.items():
        table[label], table[f"{label}_low"], table[f"{label}_high"] = (own[name][f"{CLAIMED}{end}"] for end in ("", "_low", "_high"))
    for label, found in own.items():
        table[label], table[f"{label}_low"], table[f"{label}_high"] = (found[f"{PREDICTED}{end}"] for end in ("", "_low", "_high"))
    drawn = [column for column in table.columns if not column.endswith(("_low", "_high"))]
    for poll, row in runs.polls.iterrows():
        table[poll] = row
    out.table(NAME, table)

    figure = source_bars(
        table, drawn, ranked(next(iter(runs.runs.values()))), {ACTUAL: REFERENCE, **dict.fromkeys(stands, SECONDARY), **colors(list(own))}, head=topic.turnout or (),
        labels={**{column: column for column in topic.columns}, **runs.style.labels},
        title=f"{topic.name}: {len(runs.runs)} runs side by side",
        note="whiskers are 95% bootstrap intervals over each run's subjects",
    )
    return table, [out.figure(NAME, figure)]
