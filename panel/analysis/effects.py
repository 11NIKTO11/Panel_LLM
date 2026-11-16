"""
Which of the subjects' own columns the model's answers follow: each grouping column of the
dataset tested against the per-subject option shares, by MANOVA and by PERMANOVA.

The two ask different questions and are set side by side on purpose:

- MANOVA, all columns together: what a column accounts for given the others (`partial eta2`).
- PERMANOVA, a column at a time: what it accounts for alone (`R2`), with no assumption about
  the shape of the data.

A subject is one row: repeats are averaged, and subjects with no preference or with a missing
value in a grouping column are left out.

What the subject claimed to have chosen is tested with the grouping columns, as a sanity
check: a model that never saw the claim should follow it far less than a column it was told.
"""
from typing import Dict, List, Sequence, Tuple, Union

import pandas as pd
from matplotlib.figure import Figure

from .aggregate import subject_answers
from .output import Output, save
from .plots.intervals import interval_panels
from .saved import NotApplicable, SavedRun
from .stats import manova as multivariate
from .stats import permanova as distance_based
from .stats.design import block
from .stats.compositional import DELTA

NAME = "effects"
MANOVA, PERMANOVA = "MANOVA", "PERMANOVA"
PILLAI, ETA, R2 = f"{MANOVA} {multivariate.PILLAI}", f"{MANOVA} {multivariate.ETA}", f"{PERMANOVA} {distance_based.R2}"
MANOVA_P, PERMANOVA_P = f"{MANOVA} {multivariate.P}", f"{PERMANOVA} {distance_based.P}"
SUBJECTS = "subjects"


def chance(p: float) -> str:
    """A p-value as a figure writes it."""
    return "(p<0.001)" if p < 0.001 else f"(p={p:.3f})"


def tested(run: SavedRun, predictors: Union[Sequence[str], None] = None) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    """What the tests work on: the predictors that vary, the subjects complete in them, and those subjects' option shares."""
    named = list(predictors or [*run.dataset.groups, *([run.claims.column] if run.claims and run.claims.column not in run.dataset.groups else [])])
    unknown = sorted(set(named) - set(run.subjects.columns))
    if unknown:
        raise ValueError(f"'{run.dataset.name}' has no columns {unknown}; it has {list(run.subjects.columns)}")
    varying = [column for column in named if run.subjects[column].nunique() > 1]
    if not varying:
        raise NotApplicable(f"'{run.stem}': the dataset names no grouping column that varies among the subjects")
    shares = subject_answers(run.normalized, run.topic)[run.options]
    subjects = run.subjects.loc[shares.index, varying].dropna()
    return varying, subjects, shares.loc[subjects.index]


def compute(
    run: SavedRun, predictors: Union[Sequence[str], None] = None, distance: str = "hellinger", permutations: int = 999, most: int = 1000, seed: int = 0, delta: float = DELTA,
) -> Dict[str, pd.DataFrame]:
    """
    `effects` - one row per predictor, largest partial eta squared first: `MANOVA df`, `Pillai`,
    `partial eta2`, `F`, `p`, and `PERMANOVA R2`, `F`, `p`. MANOVA tests each predictor given the
    others, so the set changes every number; by default the dataset's grouping columns and its
    claims column for the topic. PERMANOVA works on a subjects-by-subjects matrix, so above `most`
    subjects it takes a sample of that size; MANOVA always uses all. `distance` is of
    `stats.permanova.DISTANCES`.
    """
    predictors, subjects, shares = tested(run, predictors)
    several = {column: levels.several for column, levels in run.dataset.levels.items() if levels.several}
    needed = 1 + sum(block(subjects[column], several.get(column)).shape[1] for column in predictors) + len(run.options) - 1
    if len(subjects) <= needed:
        raise NotApplicable(f"'{run.stem}': {len(subjects)} complete subjects are too few to test {len(predictors)} columns ({needed} design columns and coordinates)")

    table = multivariate.manova(subjects, shares, predictors, several, delta).add_prefix(f"{MANOVA} ")
    sample = subjects.sample(most, random_state=seed).index if len(subjects) > most else subjects.index
    centred = distance_based.gower(distance_based.pairwise(shares.loc[sample].to_numpy(), distance, delta))
    for column in predictors:
        _, share, ratio, found = distance_based.permanova(centred, block(subjects.loc[sample, column], several.get(column)), permutations, seed)
        table.loc[column, [R2, f"{PERMANOVA} {distance_based.F}", f"{PERMANOVA} {distance_based.P}"]] = share, ratio, found
    table = table.sort_values(ETA, ascending=False)
    table.index.name = "column"
    return {NAME: table}


def draw(
    run: SavedRun, tables: Dict[str, pd.DataFrame], predictors: Union[Sequence[str], None] = None, distance: str = "hellinger", most: int = 1000, delta: float = DELTA,
) -> Dict[str, Figure]:
    """A panel each for Pillai's trace, the partial eta squared and the PERMANOVA R squared, a row per column, as `effects`."""
    table = tables[NAME]
    subjects = len(tested(run, predictors)[1])
    figure = interval_panels(
        table, [PILLAI, ETA, R2], percent=[ETA, R2], unit="%",
        names={PILLAI: "MANOVA Pillai's trace", ETA: "MANOVA partial eta squared", R2: "PERMANOVA R squared"},
        marks={PILLAI: table[MANOVA_P].map(chance), R2: table[PERMANOVA_P].map(chance)},
        title=f"{run.title}: what the answers follow",
        note=f"{subjects} subjects of '{run.dataset.name}'" + (f", PERMANOVA on a sample of {most}" if subjects > most else "") + f"; MANOVA: given the other columns, zero shares replaced by {delta:g}; PERMANOVA ({distance}): alone",
    )
    return {NAME: figure}


def effects(
    run: SavedRun, out: Output, predictors: Union[Sequence[str], None] = None, distance: str = "hellinger", permutations: int = 999, most: int = 1000, seed: int = 0, delta: float = DELTA,
) -> Tuple[pd.DataFrame, List[Figure]]:
    tables = compute(run, predictors, distance, permutations, most, seed, delta)
    return save(out, tables, draw(run, tables, predictors, distance, most, delta))
