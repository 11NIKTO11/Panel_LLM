"""
How far every source is from the actual result, and every run from what its subjects claimed.

Against the actual result the sources are the published polls, the claimed distribution and
each run's prediction; against the claimed one, the polls and each run's prediction. A run's
numbers are those of its own `distance` analysis, intervals included; a poll is a published
number and has no interval and no turnout.

A poll is held against the claimed distribution only when every run asked the same subjects,
so that there is one claimed distribution to hold it against.
"""
from typing import List, Tuple

import pandas as pd
from matplotlib.figure import Figure

from ..distance import NAME as DISTANCE, compute as distance
from ..estimate import high, low
from ..output import Output
from ..plots.intervals import interval_panels
from ..saved import NotApplicable
from ..sources import ACTUAL, CLAIMED, PREDICTED, TURNOUT, estimated, frame, shown, versus
from ..stats.metrics import MAE, RMSE, distances as measure
from .loaded import RunSet

NAME = "runs_distances"
AGAINST, SOURCE = "against", "source"


def distances(runs: RunSet, out: Output) -> Tuple[pd.DataFrame, List[Figure]]:
    """
    The table is indexed by what is measured against and the source: polls first, closest by MAE
    first, then the claimed, then the runs as named; one figure per reference.
    """
    topic, options = runs.topic, list(runs.topic.options)
    own = {}
    for label, run in runs.runs.items():
        try:
            own[label] = distance(run)[DISTANCE]
        except NotApplicable:
            raise NotApplicable(f"'{topic.name}' has neither claims nor a reference; there is nothing to measure a distance to")

    rows = {}
    for label, found in own.items():
        for reference in (ACTUAL, CLAIMED):
            if versus(PREDICTED, reference) in found.index:
                rows[(reference, label)] = found.loc[versus(PREDICTED, reference)]
    for label, name in runs.stands.items():
        if versus(CLAIMED, ACTUAL) in own[name].index:
            rows[(ACTUAL, label)] = own[name].loc[versus(CLAIMED, ACTUAL)]
    first = next(iter(runs.runs.values()))
    if first.reference is not None:
        actual = first.reference.loc[first.total, options].to_numpy(dtype=float)
        for poll, row in runs.polls.iterrows():
            rows[(ACTUAL, poll)] = pd.Series(measure(actual, row.to_numpy(dtype=float)))
    if len(runs.claimed) == 1:
        asked = next(iter(runs.claimed.values()))
        claimed = estimated(asked, frame(asked))[CLAIMED][options].to_numpy(dtype=float)
        for poll, row in runs.polls.iterrows():
            rows[(CLAIMED, poll)] = pd.Series(measure(claimed, row.to_numpy(dtype=float)))
    metrics = shown(bool(topic.turnout))
    columns = metrics + [end(metric) for metric in metrics for end in (low, high)]
    table = pd.DataFrame(rows).T.rename_axis([AGAINST, SOURCE]).astype(float).reindex(columns=columns)
    order = []
    for reference in (ACTUAL, CLAIMED):
        polls = [poll for poll in runs.polls.index if (reference, poll) in table.index]
        order += [(reference, poll) for poll in sorted(polls, key=lambda poll: table.loc[(reference, poll), MAE])]
        order += [(reference, source) for source in [*runs.claimed, *runs.runs] if (reference, source) in table.index]
    table = table.loc[order]
    out.table(NAME, table)

    figures = []
    for reference in table.index.get_level_values(AGAINST).unique():
        figure = interval_panels(
            table.loc[reference], metrics, per_row=3, bars=True,
            percent=[metric for metric in (MAE, RMSE, TURNOUT) if metric in metrics], zero=[TURNOUT] if topic.turnout else [],
            title=f"{topic.name}: distance from the {reference}",
            note=f"over the {len(options)} options; polls, then the claimed, then the runs; whiskers are 95% bootstrap intervals over a run's subjects"
                 + (f"; turnout: a source minus the {reference}" if topic.turnout else ""),
        )
        figures.append(out.figure(f"{NAME}_{reference}", figure))
    return table, figures
