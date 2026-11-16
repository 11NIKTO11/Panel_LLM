"""
The three things a run is read against each other as: what the model predicted, what the same
subjects claimed, and the actual result.

`frame` puts a run's answers and its subjects' claim in one table, so one resampling of
subjects moves predicted and claimed together; `distributions` reads the sources off such a
table. A source the run has nothing for (no claims map, no reference) is left out.

Two sources are a pair, a source held against a reference; `pair_distances` measures every
pair a run has, and is what the whole-run and the by-area distances both stand on.
"""
from typing import Dict, List, Sequence, Tuple, Union

import pandas as pd

from .aggregate import claimed_distribution, predicted_distribution
from .saved import SavedRun
from .stats.metrics import CORRELATION, JS, KL, MAE, RMSE, distances

PREDICTED, CLAIMED, ACTUAL = "predicted", "claimed", "actual"
SOURCES = (PREDICTED, CLAIMED, ACTUAL)
ESTIMATED = (PREDICTED, CLAIMED)
PAIRS = ((PREDICTED, ACTUAL), (PREDICTED, CLAIMED), (CLAIMED, ACTUAL))
TURNOUT = "Turnout difference"


def shown(turnout: bool) -> List[str]:
    """The measures in the order tables and figures show them: the turnout difference, if any, beside the two errors, the correlation last."""
    return [MAE, RMSE, *([TURNOUT] if turnout else []), JS, KL, CORRELATION]


def versus(source: str, reference: str) -> str:
    """A pair as tables name it."""
    return f"{source} vs {reference}"


def frame(run: SavedRun) -> pd.DataFrame:
    """One row per call: the normalized answer and, if the dataset has one, the subject's claim."""
    return run.normalized.join(run.subjects[[run.claims.column]]) if run.claims else run.normalized


def estimated(run: SavedRun, rows: pd.DataFrame) -> Dict[str, pd.Series]:
    """Predicted and, if the dataset has claims, claimed, for the given rows of `frame(run)`: what a resampling moves."""
    found = {PREDICTED: predicted_distribution(rows, run.topic, run.weighted)}
    if run.claims:
        found[CLAIMED] = claimed_distribution(rows, run.claims, run.topic)
    return found


def actual(run: SavedRun, area: Union[str, None] = None) -> Dict[str, pd.Series]:
    """The actual result of `area`, or of the whole without one; empty if the topic has no reference."""
    if run.reference is None:
        return {}
    return {ACTUAL: run.reference.loc[area or run.total, run.topic.columns].astype(float)}


def distributions(run: SavedRun, rows: pd.DataFrame, area: Union[str, None] = None) -> Dict[str, pd.Series]:
    """Source -> its distribution, for the given rows of `frame(run)`; the caller hands in the rows that belong to `area`."""
    return {**estimated(run, rows), **actual(run, area)}


def pair_distances(run: SavedRun, found: Dict[str, pd.Series], metrics: Sequence[str]) -> Dict[Tuple[str, str, str], float]:
    """
    (source, reference, measure) -> value for every pair in `found` (as `distributions` gives it), the
    turnout difference, source minus reference, last in each. `metrics` are of `stats.metrics.METRICS`
    and `TURNOUT`, which needs a topic with turnout.
    """
    options, turnout = run.options, run.topic.turnout
    over = [metric for metric in metrics if metric != TURNOUT]
    values = {}
    for source, reference in PAIRS:
        if source in found and reference in found:
            for metric, value in distances(found[reference][options].to_numpy(), found[source][options].to_numpy(), over).items():
                values[(source, reference, metric)] = value
            if TURNOUT in metrics:
                values[(source, reference, TURNOUT)] = found[source][turnout[0]] - found[reference][turnout[0]]
    return values


def ranked(run: SavedRun) -> List[str]:
    """
    The options in the order figures rank them: by the actual result of the whole, largest
    first - by the run's own prediction on a topic without a reference - with the options
    the topic's style names as last at the end.
    """
    shares = actual(run)[ACTUAL] if run.reference is not None else predicted_distribution(frame(run), run.topic, run.weighted)
    order = list(shares[run.options].sort_values(ascending=False, kind="stable").index)
    return [option for option in order if option not in run.style.last] + [option for option in order if option in run.style.last]
