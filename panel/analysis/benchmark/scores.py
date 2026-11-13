"""
How well every source says what each subject answered, and what that says of a run.

`scores` is the plain table: log-loss and Brier score per source. `knowledge` reads two things
out of the log-losses per run:

- the conditioning gain: how much better the run does with each subject's own answer than
  with its mean answer given to everybody - what the description of the subject bought;
- the knowledge ratio: how far the run's recalibrated answers get from the claimed shares
  towards the model trained on the answers. 1 is as much as that model, 0 nothing. It is
  empty when the trained model itself does no better than the claimed shares, as on a
  small sample: there is then nothing to measure a share of.

Intervals resample the subjects with every source's probabilities held as they are: neither
the trained model nor the recalibration is refitted in a draw.
"""
from typing import Callable, List, Sequence, Tuple

import pandas as pd
from matplotlib.figure import Figure

from ..bootstrap import HIGH, LOW, interval
from ..output import Output
from ..plots.intervals import interval_panels
from .ground import LOG_LOSS, MEASURES, SHARES, TRAINED, Ground, averaged, reread

SCORES, KNOWLEDGE = "benchmark_scores", "benchmark_knowledge"
GAIN, RATIO = "conditioning gain", "knowledge ratio"
SOURCE, RUN = "source", "run"


def measured(losses: pd.DataFrame, statistic: Callable[[pd.DataFrame], pd.Series], rows: Sequence[str], measures: Sequence[str], axis: str, draws: int, seed: int) -> pd.DataFrame:
    """A statistic keyed by (row, measure) as a table: a column per measure, with `_low` and `_high`."""
    edges = interval(losses, statistic, draws, seed)
    found = pd.concat({"": statistic(losses), "_low": edges[LOW], "_high": edges[HIGH]}, axis=1).unstack(level=1)
    found.columns = [f"{measure}{end}" for end, measure in found.columns]
    return found.loc[list(rows), [f"{measure}{end}" for end in ("", "_low", "_high") for measure in measures]].rename_axis(axis)


def scores(ground: Ground, out: Output, draws: int = 1000, seed: int = 0) -> Tuple[pd.DataFrame, List[Figure]]:
    """
    The log-loss and the Brier score of every source on the subjects' own answers.

    Args:
        ground: what `load_ground` made.
        out: takes the table and the figure, both named `benchmark_scores`.
        draws: bootstrap resamples of the subjects.
        seed: of the resampling.

    Returns:
        The table - a row per source: the claimed shares, the trained model, and per run its
        answers, its mean answer and its recalibrated answers; `log-loss` and `Brier`, each
        with `_low` and `_high` - and its figure.
    """
    table = out.table(SCORES, measured(ground.losses, lambda frame: frame.mean(), ground.sources, MEASURES, SOURCE, draws, seed))
    figure = interval_panels(
        table, list(MEASURES), same_scale=False,
        title=f"{ground.topic.name}: how well each source says what a subject answered",
        note=f"{len(ground.answers)} subjects, lower is better; log-loss in nats; whiskers are 95% bootstrap intervals over subjects",
    )
    return table, [out.figure(SCORES, figure)]


def knowledge(ground: Ground, out: Output, draws: int = 1000, seed: int = 0) -> Tuple[pd.DataFrame, List[Figure]]:
    """
    Per run, what the subject's description bought and how much of the trained model's
    knowledge the run holds.

    Args:
        ground: what `load_ground` made.
        out: takes the table and the figure, both named `benchmark_knowledge`.
        draws: bootstrap resamples of the subjects.
        seed: of the resampling.

    Returns:
        The table - a row per run: `conditioning gain` (nats) and `knowledge ratio`, each
        with `_low` and `_high`; the ratio and its interval are empty when the trained
        model does not beat the claimed shares - and its figure.
    """
    labels = list(ground.predicted)

    def statistic(frame: pd.DataFrame) -> pd.Series:
        loss = frame.xs(LOG_LOSS, axis=1, level=1).mean()
        found, ahead = {}, loss[SHARES] - loss[TRAINED]
        for label in labels:
            found[(label, GAIN)] = loss[averaged(label)] - loss[label]
            found[(label, RATIO)] = (loss[SHARES] - loss[reread(label)]) / ahead if ahead > 0 else float("nan")
        return pd.Series(found)

    table = measured(ground.losses, statistic, labels, (GAIN, RATIO), RUN, draws, seed)
    table.loc[table[RATIO].isna(), [f"{RATIO}_low", f"{RATIO}_high"]] = float("nan")
    out.table(KNOWLEDGE, table)
    figure = interval_panels(
        table, [GAIN, RATIO], same_scale=False,
        names={GAIN: "conditioning gain (nats)", RATIO: "knowledge ratio"},
        title=f"{ground.topic.name}: what a run knows of a subject",
        note=f"{len(ground.answers)} subjects; gain: log-loss of the mean answer minus that of the answers; ratio: 1 is as much as '{TRAINED}'",
    )
    return table, [out.figure(KNOWLEDGE, figure)]
