"""
Which options go together over subjects: the correlation and the covariance between the
options' shares.

A subject is one row: repeats are averaged first, and a subject with no preference is left
out. The shares of a subject sum to 1, so one option rising makes the others fall whatever
the model thinks: with k options and nothing else going on the correlations sit near
-1/(k-1), not at zero. Read a value against that, and a positive one as the stronger sign.
"""
from typing import List, Tuple

import pandas as pd
from matplotlib.figure import Figure

from .aggregate import subject_answers
from .output import Output
from .plots.matrices import heatmap
from .saved import SavedRun

NAME = "correlation"
COVARIANCE = "covariance"


def correlation(run: SavedRun, out: Output) -> Tuple[pd.DataFrame, List[Figure]]:
    """
    Pearson correlation and covariance between the options' shares, over subjects.

    Args:
        run: the saved run.
        out: takes the correlation table and figure, named `correlation`, and the
            covariance table and figure, named `covariance`.

    Returns:
        The correlation table - options by options; an option that never varies has no
        correlation - and the two figures, their diagonals left blank.
    """
    options = run.options
    found = subject_answers(run.normalized, run.topic)[options]
    table, covariance = found.corr(), found.cov()
    out.table(NAME, table)
    out.table(COVARIANCE, covariance)

    left = len(run.subjects) - len(found)
    note = f"{len(found)} subjects of '{run.dataset.name}'" + (f"; {left} without a preference left out" if left else "")
    figures = [
        out.figure(NAME, heatmap(
            table, bound=1, labels=run.labels,
            title=f"{run.spec.model} on {run.topic.name}: correlation between options",
            note=f"{note}; shares sum to 1, so about {-1 / (len(options) - 1):.2f} is what no relation looks like",
        )),
        out.figure(COVARIANCE, heatmap(
            covariance, decimals=4, labels=run.labels,
            title=f"{run.spec.model} on {run.topic.name}: covariance between options",
            note=note,
        )),
    ]
    return table, figures
