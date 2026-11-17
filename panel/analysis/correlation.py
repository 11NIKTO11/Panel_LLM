"""
Which options go together over subjects: the correlation and the covariance between the
options' shares.

A subject is one row: repeats are averaged first, and a subject with no preference is left
out. The shares of a subject sum to 1, so one option rising makes the others fall whatever
the model thinks: with k options and nothing else going on the correlations sit near
-1/(k-1), not at zero. Read a value against that, and a positive one as the stronger sign.
"""
from typing import Dict, List, Tuple

import pandas as pd
from matplotlib.figure import Figure

from .aggregate import subject_answers
from .output import Output, save
from .plots.matrices import heatmap
from .saved import SavedRun, pools

NAME = "correlation"
COVARIANCE = "covariance"


def compute(run: SavedRun) -> Dict[str, pd.DataFrame]:
    """
    `correlation` and `covariance` between the options' shares, over subjects: options by
    options; an option that never varies has no correlation.
    """
    pools(run, NAME)
    found = subject_answers(run.normalized, run.topic)[run.options]
    return {NAME: found.corr(), COVARIANCE: found.cov()}


def draw(run: SavedRun, tables: Dict[str, pd.DataFrame]) -> Dict[str, Figure]:
    """The two matrices as heatmaps, `correlation` and `covariance`, their diagonals left blank."""
    answered = len(subject_answers(run.normalized, run.topic))
    left = len(run.subjects) - answered
    note = f"{answered} subjects of '{run.dataset.name}'" + (f"; {left} without a preference left out" if left else "")
    return {
        NAME: heatmap(
            tables[NAME], bound=1, labels=run.labels,
            title=f"{run.title}: correlation between options",
            note=f"{note}; shares sum to 1, so about {-1 / (len(run.options) - 1):.2f} is what no relation looks like",
        ),
        COVARIANCE: heatmap(
            tables[COVARIANCE], decimals=4, labels=run.labels,
            title=f"{run.title}: covariance between options",
            note=note,
        ),
    }


def correlation(run: SavedRun, out: Output) -> Tuple[pd.DataFrame, List[Figure]]:
    tables = compute(run)
    return save(out, tables, draw(run, tables))
