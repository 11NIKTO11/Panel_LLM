"""
The three things a run is read against each other as: what the model predicted, what the same
subjects claimed, and the actual result.

`frame` puts a run's answers and its subjects' claim in one table, so one resampling of
subjects moves predicted and claimed together; `distributions` reads the sources off such a
table. A source the run has nothing for (no claims map, no reference) is left out.
"""
from typing import Dict, List, Union

import pandas as pd

from .aggregate import claimed_distribution, predicted_distribution
from .saved import SavedRun

PREDICTED, CLAIMED, ACTUAL = "predicted", "claimed", "actual"
SOURCES = (PREDICTED, CLAIMED, ACTUAL)
ESTIMATED = (PREDICTED, CLAIMED)


def frame(run: SavedRun) -> pd.DataFrame:
    """One row per call: the normalized answer and, if the dataset has one, the subject's claim."""
    return run.normalized.join(run.subjects[[run.claims.column]]) if run.claims else run.normalized


def estimated(run: SavedRun, rows: pd.DataFrame) -> Dict[str, pd.Series]:
    """Predicted and, if the dataset has claims, claimed, for the given rows of `frame(run)`: what a resampling moves."""
    found = {PREDICTED: predicted_distribution(rows, run.topic)}
    if run.claims:
        found[CLAIMED] = claimed_distribution(rows, run.claims, run.topic)
    return found


def actual(run: SavedRun, area: Union[str, None] = None) -> Dict[str, pd.Series]:
    """The actual result of `area`, or of the whole without one; empty if the topic has no reference."""
    if run.reference is None:
        return {}
    return {ACTUAL: run.reference.loc[area or run.total, run.topic.columns].astype(float)}


def distributions(run: SavedRun, rows: pd.DataFrame, area: Union[str, None] = None) -> Dict[str, pd.Series]:
    """
    Each source's distribution over the topic's columns, for the given rows of `frame(run)`;
    the caller hands in the rows that belong to `area`.
    """
    return {**estimated(run, rows), **actual(run, area)}


def ranked(run: SavedRun) -> List[str]:
    """
    The options in the order figures rank them: by the actual result of the whole, largest
    first - by the run's own prediction on a topic without a reference - with the options
    the topic's style names as last at the end.
    """
    shares = actual(run)[ACTUAL] if run.reference is not None else predicted_distribution(frame(run), run.topic)
    order = list(shares[run.options].sort_values(ascending=False, kind="stable").index)
    return [option for option in order if option not in run.style.last] + [option for option in order if option in run.style.last]
