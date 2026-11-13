"""
What every benchmark stands on: the subjects whose own answer to the topic is known, that
answer, each run's probabilities for them, and a model trained on the answers themselves.

An outcome is one of the topic's options or, on a topic with turnout, not taking part. A
run's answer becomes one distribution over the outcomes: each option's share times the
probability of taking part, and the probability of not taking part. A subject is one row,
repeats averaged.

Only subjects that every named run answered and whose own answer says something count, so
all the sources are scored on the same people.
"""
from dataclasses import dataclass
from typing import Dict, List, Sequence, Union

import numpy as np
import pandas as pd

from panel.topic import PlotStyle, Topic

from ..comparison import RunSet
from ..saved import NotApplicable, SavedRun
from ..stats.classifier import cross_validated, losses, recalibrated
from ..stats.design import indicators

SHARES, TRAINED = "claimed shares", "trained on claims"
LOG_LOSS, BRIER = "log-loss", "Brier"
MEASURES = (LOG_LOSS, BRIER)


def averaged(label: str) -> str:
    """The row of a run's mean answer, given to every subject."""
    return f"{label}: mean answer"


def reread(label: str) -> str:
    """The row of a run's answers after recalibration."""
    return f"{label}: recalibrated"


@dataclass(frozen=True)
class Ground:
    """
    Attributes:
        topic: the topic the runs answered.
        outcomes: the options, then not taking part if the topic has turnout.
        answers: each scored subject's own outcome, indexed by subject.
        predicted: run label -> subjects by outcomes, the run's probabilities.
        trained: subjects by outcomes, the cross-validated probabilities of the model
            trained on the answers.
        losses: subjects by (source, measure): the log-loss and Brier score of every
            source - the claimed shares, the trained model, and per run its answers, its
            mean answer and its recalibrated answers.
        predictors: the columns the trained model was given.
        style: the topic's colours and labels for figures.
    """
    topic: Topic
    outcomes: List[str]
    answers: pd.Series
    predicted: Dict[str, pd.DataFrame]
    trained: pd.DataFrame
    losses: pd.DataFrame
    predictors: List[str]
    style: PlotStyle

    @property
    def sources(self) -> List[str]:
        """Every scored source, in the order of the tables."""
        return list(dict.fromkeys(self.losses.columns.get_level_values(0)))


def outcome_shares(run: SavedRun) -> pd.DataFrame:
    """
    One row per subject, a column per outcome, summing to 1; a subject is left out if no
    answer of it says which option while saying it might take part.
    """
    frame, options = run.normalized, run.options
    if not run.topic.turnout:
        return frame[options].dropna().groupby(level=0, sort=False).mean()
    taking, staying = run.topic.turnout
    defined = frame[options].notna().all(axis=1) | (frame[taking] == 0)
    joint = frame[options].fillna(0.0).mul(frame[taking], axis=0).assign(**{staying: frame[staying]})
    return joint[defined].dropna().groupby(level=0, sort=False).mean()


def own_answers(run: SavedRun) -> pd.Series:
    """Each subject's own outcome; a subject whose answer says nothing is left out."""
    meaning = dict(run.claims.options)
    if run.topic.turnout:
        meaning.update({answer: run.topic.turnout[1] for answer in run.claims.abstained})
    return run.subjects[run.claims.column].map(meaning).dropna()


def load_ground(runs: RunSet, predictors: Union[Sequence[str], None] = None, strength: float = 0.3, folds: int = 5, seed: int = 0) -> Ground:
    """
    Fit the trained model and score every source on the subjects all runs share.

    Args:
        runs: the runs to benchmark.
        predictors: the columns the trained model is given; the dataset's grouping columns
            by default.
        strength: the trained logit's `C`, the inverse of its ridge penalty.
        folds: of both cross-validations (the trained model, the recalibration).
        seed: of the folds.

    Raises:
        NotApplicable: if the dataset has no answers to the topic, the runs are of
            different datasets, or too few subjects are left.
        ValueError: on a predictor the dataset does not have.
    """
    first = next(iter(runs.runs.values()))
    if not first.claims:
        raise NotApplicable(f"'{first.dataset.name}' has no answers to '{runs.topic.name}' to train on or score against")
    if len({run.dataset.name for run in runs.runs.values()}) > 1:
        raise NotApplicable("the runs are of different datasets; a benchmark needs the same subjects")
    predictors = list(predictors or [column for column in first.dataset.groups if column != first.claims.column])
    unknown = sorted(set(predictors) - set(first.subjects.columns))
    if unknown:
        raise ValueError(f"'{first.dataset.name}' has no columns {unknown}; it has {list(first.subjects.columns)}")

    outcomes = first.options + ([runs.topic.turnout[1]] if runs.topic.turnout else [])
    predicted = {label: outcome_shares(run) for label, run in runs.runs.items()}
    answers = own_answers(first)
    for shares in predicted.values():
        answers = answers[answers.index.isin(shares.index)]
    if len(answers) < 2 * folds or answers.nunique() < 2:
        raise NotApplicable(f"{len(answers)} subjects with a known answer in every run are too few to train on")
    predicted = {label: shares.loc[answers.index, outcomes] for label, shares in predicted.items()}

    levels = first.dataset.levels
    design = np.hstack([
        indicators(first.subjects.loc[answers.index, column], levels[column].several if column in levels else None, levels[column].bands if column in levels else None)
        for column in predictors
    ])
    chosen = answers.to_numpy()
    answered = pd.get_dummies(answers).reindex(columns=outcomes, fill_value=0).to_numpy(dtype=float)
    trained = pd.DataFrame(cross_validated(design, chosen, outcomes, strength, folds, seed), index=answers.index, columns=outcomes)

    everyone = lambda row: np.tile(np.asarray(row, dtype=float), (len(answers), 1))
    sources = {SHARES: everyone(answered.mean(axis=0)), TRAINED: trained.to_numpy()}
    for label, shares in predicted.items():
        sources[label] = shares.to_numpy()
        sources[averaged(label)] = everyone(shares.mean())
        sources[reread(label)] = recalibrated(shares.to_numpy(), chosen, outcomes, folds, seed)
    scored = {(source, measure): values for source, probabilities in sources.items() for measure, values in zip(MEASURES, losses(probabilities, answered))}
    return Ground(
        topic=runs.topic, outcomes=outcomes, answers=answers, predicted=predicted, trained=trained,
        losses=pd.DataFrame(scored, index=answers.index), predictors=predictors, style=runs.style,
    )


def describe(ground: Ground) -> str:
    """What the benchmark stands on, in a few lines."""
    counts = ground.answers.value_counts().reindex(ground.outcomes, fill_value=0)
    return "\n".join([
        f"{ground.topic.name}: {len(ground.answers)} subjects with a known answer in every run",
        f"  runs       {', '.join(ground.predicted)}",
        f"  trained on {', '.join(ground.predictors)}",
        f"  answers    {', '.join(f'{outcome} {count}' for outcome, count in counts.items())}",
    ])
