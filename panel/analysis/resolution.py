"""
How finely the model tells subjects apart: which numbers it writes at all, and how many
different answers it gives.

Both read the answers as saved, not normalised. The numbers are counted over every call, since
they are what the model wrote. The answers are counted within one repeat - one pass over the
subjects, an answer each: two subjects have the same answer when every topic column matches. A
run that repeats gives a count per repeat; their mean is drawn, with the least and the most. Counted
over all calls instead, the model's noise between repeats of one prompt would pass for telling
subjects apart, and averaged per subject first, the numbers it rounds to would be gone.
"""
from typing import Dict, List, Tuple

import pandas as pd
from matplotlib.figure import Figure

from panel.run.record import DELIVERED
from panel.run.resolve import REPEAT

from .estimate import high, low
from .output import Output, save
from .plots.curves import cumulative
from .plots.distributions import spikes
from .plots.style import REFERENCE
from .saved import NotApplicable, SavedRun
from .sources import ranked

NAME = "resolution"
VALUES = "resolution_values"
ANSWERS = "resolution_answers"
COLUMNS = "resolution_columns"
COLUMN, VALUE, COUNT, SHARE = "column", "value", "answers", "share"
DISTINCT = "values"
ORDINALS = ("first", "second", "third")
PROMPTS = "prompts"
EXPECTED, ACTUAL = "expected", "actual"
EXACT = 1e-9


def _covered(frame: pd.DataFrame) -> pd.Series:
    """The share of rows the most common distinct rows cover, by how many of them are taken."""
    covered = frame.value_counts(normalize=True, dropna=False).cumsum().reset_index(drop=True)
    covered.index += 1
    return covered


def compute(run: SavedRun) -> Dict[str, pd.DataFrame]:
    """
    `resolution` - one row per topic column: how many different values it holds, and the three most
    common with the share of answers holding each. `resolution_values` - every value of every column
    with its count and share. `resolution_answers` - the share of subjects the most common distinct
    answers cover, by how many are taken: the mean over the repeats, with the least and the most
    when the run repeats, and the same share for the subjects' prompts.
    """
    if DELIVERED in run.record:
        raise NotApplicable("the run is kept as a table of means, not as the answers given")
    columns = run.topic.columns
    values = pd.concat([
        run.results[column].value_counts().rename(COUNT).rename_axis(VALUE).reset_index().assign(**{COLUMN: column})
        for column in columns
    ], ignore_index=True)[[COLUMN, VALUE, COUNT]]
    values[SHARE] = values[COUNT] / len(run.results)

    rows = {}
    for column, found in values.groupby(COLUMN, sort=False):
        row = {DISTINCT: len(found)}
        for ordinal, (_, top) in zip(ORDINALS, found.head(len(ORDINALS)).iterrows()):
            row[ordinal], row[f"{ordinal}_{SHARE}"] = top[VALUE], top[SHARE]
        rows[column] = row

    passes = [part for _, part in run.results.groupby(REPEAT)] if REPEAT in run.results else [run.results]
    reached = pd.concat([_covered(part[columns]) for part in passes], axis=1).fillna(1.0)
    covered = {COUNT: reached.mean(axis=1)}
    if len(passes) > 1:
        covered[low(COUNT)], covered[high(COUNT)] = reached.min(axis=1), reached.max(axis=1)
    if run.prompts is not None:
        covered[PROMPTS] = _covered(run.prompts.groupby(level=0, sort=False).first().to_frame())
    return {NAME: pd.DataFrame.from_dict(rows, orient="index"), VALUES: values, ANSWERS: pd.DataFrame(covered).rename_axis("distinct")}


def draw(run: SavedRun, tables: Dict[str, pd.DataFrame]) -> Dict[str, Figure]:
    """
    A stem per value of every column, `resolution`; the subjects covered by the most common
    answers, `resolution_answers`; and the answers covered by a column's most common values,
    `resolution_columns`: a line per option and a dotted one for turnout - its first column, the
    other being its mirror.
    """
    topic, calls, subjects = run.topic, len(run.results), len(run.subjects)
    values = dict(tuple(tables[VALUES].groupby(COLUMN, sort=False)))
    covered = tables[ANSWERS]
    answers = {label: covered[name].dropna() for name, label in ((PROMPTS, EXPECTED), (COUNT, ACTUAL)) if name in covered}
    asked = f"{len(answers[EXPECTED])} prompts, " if EXPECTED in answers else ""
    most = len(answers[ACTUAL])
    band = (covered[low(COUNT)].dropna(), covered[high(COUNT)].dropna()) if low(COUNT) in covered else None
    least = int((band[1] < 1 - EXACT).sum()) + 1 if band else most
    given = str(most) if least == most else f"{least}-{most}"
    repeated = f", the mean of {run.spec.repeats} repeats, shaded from the least to the most" if band else ""
    turnout = list(topic.turnout[:1]) if topic.turnout else []
    columns = {
        **({EXPECTED: _covered(run.prompts.to_frame())} if run.prompts is not None else {}),
        **{run.labels[column]: values[column].set_index(pd.RangeIndex(1, len(values[column]) + 1))[SHARE].cumsum() for column in ranked(run) + turnout},
    }
    return {
        NAME: spikes(
            {column: found.set_index(VALUE)[SHARE] for column, found in values.items()},
            ranked(run), topic.turnout or (), run.style.colors, run.labels,
            title=f"{run.title}: the numbers the model writes",
            note=f"{calls} answers as saved; a stem is as tall as the share of answers holding exactly that value, the most common is named",
        ),
        ANSWERS: cumulative(
            answers, EXPECTED if EXPECTED in answers else None, {ACTUAL: REFERENCE},
            bands={ACTUAL: band} if band else None, counts={ACTUAL: given},
            xlabel="how many of the most common answers are taken", ylabel=f"share of the {subjects} subjects they cover",
            title=f"{run.title}: {asked}{given} answers",
            note=f"{subjects} subjects{repeated}; expected: an answer of its own for every different prompt",
        ),
        COLUMNS: cumulative(
            columns, EXPECTED if EXPECTED in columns else None,
            {run.labels[column]: color for column, color in run.style.colors.items()}, [run.labels[column] for column in turnout],
            xlabel="how many of the most common values are taken", ylabel=f"share of the {calls} answers they cover",
            title=f"{run.title}: different values per column",
            note=f"{calls} answers as saved; expected: a value of its own for every different prompt; in brackets how many different; dotted: turnout",
        ),
    }


def resolution(run: SavedRun, out: Output) -> Tuple[pd.DataFrame, List[Figure]]:
    tables = compute(run)
    return save(out, tables, draw(run, tables))
