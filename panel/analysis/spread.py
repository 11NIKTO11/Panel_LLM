"""
How much the subjects differ in what the model says of them: each column's distribution over
subjects, and how far each subject's answer sits from the mean answer.

A subject is one row here: repeats are averaged first, so this is the spread between subjects
and not the model's noise on one of them. A subject with no preference in any answer has no
distribution over the options and is left out; the figures say how many.
"""
from typing import List, Tuple

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .aggregate import subject_answers
from .output import Output
from .plots.distributions import histograms, violins
from .saved import SavedRun
from .sources import ranked

NAME = "spread"
DISTANCES = "spread_distances"
EUCLIDEAN, COSINE = "Euclidean distance from the mean answer", "Cosine distance from the mean answer"
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def spread(run: SavedRun, out: Output) -> Tuple[pd.DataFrame, List[Figure]]:
    """
    The distribution of every topic column over subjects, and the subjects' distances from the mean.

    Args:
        run: the saved run.
        out: takes the table and the violin figure, named `spread`, and a second table and
            the histogram figure, named `spread_distances`: per subject, the Euclidean and
            the cosine distance of its option shares from their mean over subjects.

    Returns:
        The table - one row per topic column: mean, standard deviation, minimum, the 5th,
        25th, 50th, 75th and 95th percentile, maximum - and the two figures.
    """
    topic, options = run.topic, run.options
    found = subject_answers(run.normalized, topic)
    table = found.describe(percentiles=QUANTILES).T.drop(columns="count")
    out.table(NAME, table)

    shares = found[options].to_numpy()
    mean = shares.mean(axis=0)
    distances = pd.DataFrame({
        EUCLIDEAN: np.linalg.norm(shares - mean, axis=1),
        COSINE: 1 - shares @ mean / (np.linalg.norm(shares, axis=1) * np.linalg.norm(mean)),
    }, index=found.index)
    out.table(DISTANCES, distances)

    left = len(run.subjects) - len(found)
    note = f"{len(found)} subjects of '{run.dataset.name}'" + (f"; {left} without a preference left out" if left else "")
    figures = [
        out.figure(NAME, violins(
            found, ranked(run), topic.turnout or (), run.style.colors, run.labels,
            title=f"{run.spec.model} on {topic.name}: answers per subject",
            note=f"{note}; the dot is the median, the bar the middle half",
        )),
        out.figure(DISTANCES, histograms(
            distances, "subjects",
            title=f"{run.spec.model} on {topic.name}: how far subjects are from the mean answer",
            note=f"{note}; over the {len(options)} options",
        )),
    ]
    return table, figures
