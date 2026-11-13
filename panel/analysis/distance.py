"""
How far the sources are from each other over the whole run: predicted from actual, predicted
from claimed, claimed from actual.

Each pair is a source held against a reference. The intervals come from one resampling of
subjects for all pairs, so predicted and claimed move together, as they do in truth: both are
about the same subjects.
"""
from typing import List, Tuple

import pandas as pd
from matplotlib.figure import Figure

from .bootstrap import HIGH, LOW, interval
from .metrics import CORRELATION, JS, KL, MAE, RMSE, distances
from .output import Output
from .plots.intervals import interval_panels
from .saved import NotApplicable, SavedRun
from .sources import ACTUAL, CLAIMED, PREDICTED, actual, estimated, frame

NAME = "distance"
PAIRS = ((PREDICTED, ACTUAL), (PREDICTED, CLAIMED), (CLAIMED, ACTUAL))
TURNOUT = "Turnout difference"


def shown(turnout: bool) -> List[str]:
    """The measures in the order tables and figures show them: the turnout difference, if any, beside the two errors, the correlation last."""
    return [MAE, RMSE, *([TURNOUT] if turnout else []), JS, KL, CORRELATION]


def distance(run: SavedRun, out: Output, draws: int = 1000, seed: int = 0, level: float = 0.95) -> Tuple[pd.DataFrame, List[Figure]]:
    """
    Every metric of `metrics` for each pair of sources the run has, with bootstrap intervals.

    Args:
        run: the saved run.
        out: takes the table and the figure, both named `distance`.
        draws, seed, level: of the bootstrap intervals.

    Returns:
        The table - one row per pair `<source> vs <reference>`, a column per metric over the
        options, `Turnout difference` (source minus reference) on a topic with turnout, and
        `<metric>_low`, `<metric>_high` - and its figure.

    Raises:
        NotApplicable: if the run has neither claims nor a reference, so no pair.
    """
    options, turnout = run.options, run.topic.turnout
    counted = actual(run)

    def measure(rows: pd.DataFrame) -> pd.Series:
        found = {**estimated(run, rows), **counted}
        values = {}
        for source, reference in PAIRS:
            if source in found and reference in found:
                pair = f"{source} vs {reference}"
                for metric, value in distances(found[reference][options].to_numpy(), found[source][options].to_numpy()).items():
                    values[(pair, metric)] = value
                if turnout:
                    values[(pair, TURNOUT)] = found[source][turnout[0]] - found[reference][turnout[0]]
        return pd.Series(values, dtype=float)

    measured = measure(frame(run))
    if measured.empty:
        raise NotApplicable(f"'{run.stem}' has neither claims nor a reference; there is nothing to measure a distance to")
    edges = interval(frame(run), measure, draws, seed, level)
    metrics = shown(bool(turnout))
    pairs = list(dict.fromkeys(pair for pair, _ in measured.index))
    table = measured.unstack().loc[pairs, metrics]
    for metric in metrics:
        table[f"{metric}_low"] = edges[LOW].unstack().loc[pairs, metric]
        table[f"{metric}_high"] = edges[HIGH].unstack().loc[pairs, metric]
    out.table(NAME, table)

    figure = interval_panels(
        table, metrics, per_row=3, percent=[metric for metric in (MAE, RMSE, TURNOUT) if metric in metrics], zero=[TURNOUT] if turnout else [], bars=True,
        title=f"{run.spec.model} on {run.topic.name}: distances between sources",
        note=f"{len(run.subjects)} subjects of '{run.dataset.name}', over the {len(options)} options; whiskers are {level:.0%} bootstrap intervals over subjects",
    )
    return table, [out.figure(NAME, figure)]
