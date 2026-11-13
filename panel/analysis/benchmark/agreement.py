"""
Whether a run moves an outcome's probability for the same subjects as the trained model
does, and by as much.

Per run and outcome, over the subjects: the correlation of the run's probability with the
trained model's, and the ratio of their standard deviations - above 1 the run spreads the
subjects further apart on that outcome than the answers support, below 1 less.
"""
from typing import List, Tuple

import pandas as pd
from matplotlib.figure import Figure

from ..output import Output
from ..plots.matrices import grid
from ..plots.style import EFFECTS
from .ground import TRAINED, Ground

NAME = "benchmark_agreement"
CORRELATION, SPREAD = "correlation", "spread ratio"
RUN, OUTCOME = "run", "outcome"


def agreement(ground: Ground, out: Output) -> Tuple[pd.DataFrame, List[Figure]]:
    """
    Each run's probabilities against the trained model's, outcome by outcome.

    Args:
        ground: what `load_ground` made.
        out: takes the table, named `benchmark_agreement`, and two heatmaps, outcomes by
            runs: `benchmark_agreement_correlation` and `benchmark_agreement_spread`.

    Returns:
        The table - indexed by run and outcome: `correlation`, `spread ratio`; empty where
        a side does not vary - and the two figures.
    """
    found = {
        label: pd.DataFrame({CORRELATION: shares.corrwith(ground.trained), SPREAD: shares.std() / ground.trained.std().where(ground.trained.std() > 0)})
        for label, shares in ground.predicted.items()
    }
    table = out.table(NAME, pd.concat(found, names=[RUN, OUTCOME]))
    labels = {outcome: ground.style.labels.get(outcome, outcome) for outcome in ground.outcomes}
    note = f"{len(ground.answers)} subjects; against '{TRAINED}', over the subjects"
    figures = [
        out.figure(f"{NAME}_correlation", grid(
            table[CORRELATION].unstack(RUN).loc[ground.outcomes, list(found)].clip(lower=0).rename(index=labels), cmap=EFFECTS,
            title=f"{ground.topic.name}: does a run raise an outcome for the same subjects", note=note + "; correlation, negative shown as 0",
        )),
        out.figure(f"{NAME}_spread", grid(
            table[SPREAD].unstack(RUN).loc[ground.outcomes, list(found)].rename(index=labels), cmap=EFFECTS,
            title=f"{ground.topic.name}: how far apart a run puts the subjects", note=note + "; ratio of standard deviations, 1 is as far as the trained model",
        )),
    ]
    return table, figures
