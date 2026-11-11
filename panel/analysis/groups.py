"""
The run group by group: what the model predicts for the subjects at each level of every
grouping column the dataset names, what those subjects claimed, and how much they differ
among themselves.

The levels of a column, their order, the bands of a numeric one and whether a subject may be
at several are the dataset's to declare (`levels.members`). A level is only as good as its
subjects are many, so their count comes with every row.
"""
from typing import List, Tuple

import pandas as pd
from matplotlib.figure import Figure

from .aggregate import subject_answers
from .levels import members
from .output import Output
from .plots.stacked import stacked_bars
from .saved import NotApplicable, SavedRun
from .sources import CLAIMED, PREDICTED, estimated, frame

NAME = "groups"
CLAIMS, SPREAD = "groups_claimed", "groups_spread"
GROUP, LEVEL, SUBJECTS = "group", "level", "subjects"


def groups(run: SavedRun, out: Output) -> Tuple[pd.DataFrame, List[Figure]]:
    """
    The predicted distribution at each level of each grouping column.

    Args:
        run: the saved run.
        out: takes the table, named `groups`; `groups_claimed`, the same rows for what the
            subjects claimed, if the dataset has claims; `groups_spread`, the standard
            deviation of each column over the level's subjects; and a figure per grouping
            column, named `groups_<column>`.

    Returns:
        The table - indexed by grouping column and level, levels in the order `levels.members` gives:
        `subjects`, then the topic's columns as `predicted_distribution` gives them - and
        the figures, the option shares stacked per level.

    Raises:
        NotApplicable: if the dataset names no grouping columns.
    """
    topic, options = run.topic, run.options
    if not run.dataset.groups:
        raise NotApplicable(f"'{run.stem}': the dataset names no grouping columns")
    rows, answers = frame(run), subject_answers(run.normalized, topic)
    predicted, claimed, spread, figures = {}, {}, {}, []
    for column in run.dataset.groups:
        declared = run.dataset.levels.get(column)
        for name, at in members(run.subjects[column], declared).items():
            count = len(at)
            found = estimated(run, rows.loc[rows.index.isin(at)])
            predicted[(column, name)] = pd.concat([pd.Series({SUBJECTS: count}), found[PREDICTED]])
            if CLAIMED in found:
                claimed[(column, name)] = pd.concat([pd.Series({SUBJECTS: count}), found[CLAIMED]])
            spread[(column, name)] = pd.concat([pd.Series({SUBJECTS: count}), answers.loc[answers.index.isin(at)].std()])
        shown = pd.DataFrame({name: row[options] for (group, name), row in predicted.items() if group == column}).T
        shown.index = [f"{name}  (n={int(predicted[(column, name)][SUBJECTS])})" for name in shown.index]
        figures.append(out.figure(f"{NAME}_{column}", stacked_bars(
            shown, run.style.colors, run.labels,
            title=f"{run.spec.model} on {topic.name}: predicted by {column}",
            note=f"{len(run.subjects)} subjects of '{run.dataset.name}'; shares of those predicted to vote" + ("; a subject counts at each level it holds" if declared and declared.several else ""),
        )))

    def table(found: dict) -> pd.DataFrame:
        return pd.DataFrame(found).T.rename_axis([GROUP, LEVEL]).astype({SUBJECTS: int})

    out.table(SPREAD, table(spread))
    if claimed:
        out.table(CLAIMS, table(claimed))
    return out.table(NAME, table(predicted)), figures
