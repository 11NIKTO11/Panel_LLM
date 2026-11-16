"""
How much the subjects differ in what the model says of them: each column's distribution over
subjects, and how far each subject's answer sits from the mean answer.

A subject is one row here: repeats are averaged first, so this is the spread between subjects
and not the model's noise on one of them. A subject with no preference in any answer has no
distribution over the options and is left out; the figures say how many.
"""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .aggregate import subject_answers
from .output import Output, save
from .plots.distributions import histograms, violins
from .saved import SavedRun
from .sources import ranked

NAME = "spread"
DISTANCES = "spread_distances"
EUCLIDEAN, COSINE = "Euclidean distance from the mean answer", "Cosine distance from the mean answer"
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def compute(run: SavedRun) -> Dict[str, pd.DataFrame]:
    """
    `spread` - one row per topic column: mean, standard deviation, minimum, the percentiles of
    `QUANTILES`, maximum - and `spread_distances`: per subject, the Euclidean and the cosine
    distance of its option shares from their mean over subjects.
    """
    found = subject_answers(run.normalized, run.topic)
    shares = found[run.options].to_numpy()
    mean = shares.mean(axis=0)
    distances = pd.DataFrame({
        EUCLIDEAN: np.linalg.norm(shares - mean, axis=1),
        COSINE: 1 - shares @ mean / (np.linalg.norm(shares, axis=1) * np.linalg.norm(mean)),
    }, index=found.index)
    return {NAME: found.describe(percentiles=QUANTILES).T.drop(columns="count"), DISTANCES: distances}


def draw(run: SavedRun, tables: Dict[str, pd.DataFrame]) -> Dict[str, Figure]:
    """A violin per topic column over the subjects, `spread`, and the two distances as histograms, `spread_distances`."""
    topic = run.topic
    found = subject_answers(run.normalized, topic)
    left = len(run.subjects) - len(found)
    note = f"{len(found)} subjects of '{run.dataset.name}'" + (f"; {left} without a preference left out" if left else "")
    return {
        NAME: violins(
            found, ranked(run), topic.turnout or (), run.style.colors, run.labels,
            title=f"{run.title}: answers per subject",
            note=f"{note}; the dot is the median, the bar the middle half",
        ),
        DISTANCES: histograms(
            tables[DISTANCES], "subjects",
            title=f"{run.title}: how far subjects are from the mean answer",
            note=f"{note}; over the {len(run.options)} options",
        ),
    }


def spread(run: SavedRun, out: Output) -> Tuple[pd.DataFrame, List[Figure]]:
    tables = compute(run)
    return save(out, tables, draw(run, tables))
