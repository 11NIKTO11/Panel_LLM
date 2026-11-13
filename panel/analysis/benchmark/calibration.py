"""
Whether a probability means what it says: of everything a source gave about 20 %, did about
a fifth happen.

Every subject-and-outcome pair is one probability; they are pooled over the outcomes, sorted
into bins by size, and each bin's mean probability is set against how often its outcome was
the subject's own answer.
"""
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..output import Output
from ..plots.curves import curves
from ..plots.style import REFERENCE, colors
from .ground import TRAINED, Ground

NAME = "benchmark_calibration"
PREDICTED, OBSERVED, PAIRS = "predicted", "observed", "pairs"
SOURCE, BIN = "source", "bin"
EDGES = (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)


def calibration(ground: Ground, out: Output, edges: Sequence[float] = EDGES) -> Tuple[pd.DataFrame, List[Figure]]:
    """
    The calibration curve of the trained model and of every run.

    Args:
        ground: what `load_ground` made.
        out: takes the table and the figure, both named `benchmark_calibration`.
        edges: the bins' edges, from 0 to 1.

    Returns:
        The table - indexed by source and bin: the bin's mean `predicted` probability, the
        `observed` frequency and how many `pairs` it holds; empty bins left out - and its figure.
    """
    answered = pd.get_dummies(ground.answers).reindex(columns=ground.outcomes, fill_value=0).to_numpy(dtype=float).ravel()
    names = [f"{low:g}-{high:g}" for low, high in zip(edges[:-1], edges[1:])]
    found = {}
    for source, shares in {TRAINED: ground.trained, **ground.predicted}.items():
        given = shares[ground.outcomes].to_numpy(dtype=float).ravel()
        at = np.digitize(given, list(edges[1:-1]))
        pairs = pd.DataFrame({PREDICTED: given, OBSERVED: answered}).groupby(at)
        found[source] = pairs.mean().assign(**{PAIRS: pairs.size()}).rename(index=dict(enumerate(names)))
    table = out.table(NAME, pd.concat(found, names=[SOURCE, BIN]))
    figure = curves(
        {source: table.loc[source] for source in found}, PREDICTED, OBSERVED,
        colors={TRAINED: REFERENCE, **colors(ground.predicted)}, diagonal=True,
        xlabel="predicted probability (mean of the bin)", ylabel="how often it was the answer",
        title=f"{ground.topic.name}: calibration, pooled over the outcomes",
        note=f"{len(ground.answers)} subjects by {len(ground.outcomes)} outcomes; on the dotted line a probability means what it says",
    )
    return table, [out.figure(NAME, figure)]
