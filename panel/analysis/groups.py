"""
The run group by group: what the model predicts for the subjects at each level of every
grouping column the dataset names, what those subjects claimed, and how much they differ
among themselves.

The levels of a column, their order, the bands of a numeric one and whether a subject may be
at several are the dataset's to declare (`levels.members`). A level is only as good as its
subjects are many, so their count comes with every row.
"""
from typing import Dict, List, Tuple

import pandas as pd
from matplotlib.figure import Figure

from .aggregate import subject_answers
from .levels import members
from .output import Output, save
from .plots.stacked import stacked_bars
from .saved import NotApplicable, SavedRun
from .sources import CLAIMED, PREDICTED, estimated, frame

NAME = "groups"
CLAIMS, SPREAD = "groups_claimed", "groups_spread"
GROUP, LEVEL, SUBJECTS = "group", "level", "subjects"


def compute(run: SavedRun) -> Dict[str, pd.DataFrame]:
    """
    `groups` - indexed by grouping column and level, levels in the order `levels.members` gives:
    `subjects`, then the topic's columns as `predicted_distribution` gives them; `groups_spread`,
    the standard deviation of each column over the level's subjects; and `groups_claimed`, the
    rows of `groups` for what the subjects claimed, if the dataset has claims.
    """
    if not run.dataset.groups:
        raise NotApplicable(f"'{run.stem}': the dataset names no grouping columns")
    rows, answers = frame(run), subject_answers(run.normalized, run.topic)
    predicted, claimed, spread = {}, {}, {}
    for column in run.dataset.groups:
        for name, at in members(run.subjects[column], run.dataset.levels.get(column)).items():
            count = pd.Series({SUBJECTS: len(at)})
            found = estimated(run, rows.loc[rows.index.isin(at)])
            predicted[(column, name)] = pd.concat([count, found[PREDICTED]])
            if CLAIMED in found:
                claimed[(column, name)] = pd.concat([count, found[CLAIMED]])
            spread[(column, name)] = pd.concat([count, answers.loc[answers.index.isin(at)].std()])

    def table(found: dict) -> pd.DataFrame:
        return pd.DataFrame(found).T.rename_axis([GROUP, LEVEL]).astype({SUBJECTS: int})

    return {NAME: table(predicted), SPREAD: table(spread), **({CLAIMS: table(claimed)} if claimed else {})}


def draw(run: SavedRun, tables: Dict[str, pd.DataFrame]) -> Dict[str, Figure]:
    """A figure per grouping column, named `groups_<column>`: the option shares stacked per level."""
    table, figures = tables[NAME], {}
    for column in run.dataset.groups:
        declared = run.dataset.levels.get(column)
        levels = table.loc[column]
        shown = levels[run.options].rename(index=lambda name: f"{name}  (n={int(levels.loc[name, SUBJECTS])})")
        figures[f"{NAME}_{column}"] = stacked_bars(
            shown, run.style.colors, run.labels,
            title=f"{run.title}: predicted by {column}",
            note=f"{len(run.subjects)} subjects of '{run.dataset.name}'; shares of those predicted to vote" + ("; a subject counts at each level it holds" if declared and declared.several else ""),
        )
    return figures


def groups(run: SavedRun, out: Output) -> Tuple[pd.DataFrame, List[Figure]]:
    tables = compute(run)
    return save(out, tables, draw(run, tables))
