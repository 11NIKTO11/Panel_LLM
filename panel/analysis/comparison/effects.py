"""
Which of the subjects' columns each run's answers follow, the runs side by side.

A run's numbers are those of its own `effects` analysis.
"""
from typing import List, Sequence, Tuple, Union

import pandas as pd
from matplotlib.figure import Figure

from ..effects import ETA, MANOVA_P, PERMANOVA_P, PILLAI, R2, chance, effects as single
from ..output import Output
from ..plots.matrices import grid
from ..plots.style import EFFECTS
from ..stats.compositional import DELTA
from .loaded import RunSet

NAME = "runs_effects"
RUN, COLUMN = "run", "column"


def effects(runs: RunSet, out: Output, predictors: Union[Sequence[str], None] = None, delta: float = DELTA) -> Tuple[pd.DataFrame, List[Figure]]:
    """
    Each grouping column's MANOVA and PERMANOVA effect, per run.

    Args:
        runs: the runs to compare.
        predictors: the columns to test, as the single-run `effects` takes them.
        delta: what a zero share is replaced by before a log-ratio transform.
        out: takes the table, named `runs_effects`, and three heatmaps, columns by runs:
            `runs_effects_pillai` and `runs_effects_r2` with p-values, `runs_effects_eta2`.

    Returns:
        The table - indexed by run and grouping column, the columns of the single-run
        `effects` - and the two figures.

    Raises:
        NotApplicable: if a run has nothing to test.
    """
    quiet = Output(out.store, out.stem, save=False)
    own = {label: single(run, quiet, predictors, delta=delta)[0] for label, run in runs.runs.items()}
    table = pd.concat(own, names=[RUN, COLUMN])
    out.table(NAME, table)

    order = table[ETA].groupby(level=COLUMN).mean().sort_values(ascending=False).index

    def wide(found: pd.Series) -> pd.DataFrame:
        return found.unstack(RUN).reindex(index=order, columns=list(own))

    figures = [
        out.figure(f"{NAME}_pillai", grid(
            wide(table[PILLAI]), below=wide(table[MANOVA_P].map(chance)), cmap=EFFECTS,
            title=f"{runs.topic.name}: MANOVA Pillai's trace",
            note="per run, each column given the others; its p-value in brackets",
        )),
        out.figure(f"{NAME}_eta2", grid(
            wide(table[ETA]), decimals=1, scale=100, cmap=EFFECTS,
            title=f"{runs.topic.name}: MANOVA partial eta squared (%)",
            note="share of the variation between subjects a column accounts for, given the other columns",
        )),
        out.figure(f"{NAME}_r2", grid(
            wide(table[R2]), below=wide(table[PERMANOVA_P].map(chance)), decimals=1, scale=100, cmap=EFFECTS,
            title=f"{runs.topic.name}: PERMANOVA R squared (%)",
            note="share of the distances between subjects a column accounts for alone; its permutation p-value in brackets",
        )),
    ]
    return table, figures
