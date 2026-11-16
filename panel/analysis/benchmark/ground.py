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
from typing import Dict, List, Mapping, Sequence, Union

import numpy as np
import pandas as pd

from panel.topic import PlotStyle, Topic

from ..aggregate import outcome_shares
from ..levels import banded
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
        outcomes: the options, then not taking part if the topic has turnout.
        answers: each scored subject's own outcome, indexed by subject.
        predicted: run label -> subjects by outcomes, the run's probabilities.
        trained: subjects by outcomes, cross-validated.
        losses: subjects by (source, measure); the sources are the claimed shares, the trained
            model, and per run its answers, its mean answer and its recalibrated answers.
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


def own_answers(run: SavedRun) -> pd.Series:
    """Each subject's own outcome; a subject whose answer says nothing is left out."""
    meaning = dict(run.claims.options)
    if run.topic.turnout:
        meaning.update({answer: run.topic.turnout[1] for answer in run.claims.abstained})
    return run.subjects[run.claims.column].map(meaning).dropna()


def load_ground(runs: Mapping[str, SavedRun], predictors: Union[Sequence[str], None] = None, strength: float = 0.3, folds: int = 5, seed: int = 0) -> Ground:
    """
    The runs are of one topic; `predictors` are the dataset's grouping columns by default;
    `strength` is the trained logit's `C`, the inverse of its ridge penalty; `folds` and `seed`
    are of both cross-validations (the trained model, the recalibration).
    """
    first = next(iter(runs.values()))
    topic = first.topic
    if not first.claims:
        raise NotApplicable(f"'{first.dataset.name}' has no answers to '{topic.name}' to train on or score against")
    if len({run.dataset.name for run in runs.values()}) > 1:
        raise NotApplicable("the runs are of different datasets; a benchmark needs the same subjects")
    predictors = list(predictors or [column for column in first.dataset.groups if column != first.claims.column])
    unknown = sorted(set(predictors) - set(first.subjects.columns))
    if unknown:
        raise ValueError(f"'{first.dataset.name}' has no columns {unknown}; it has {list(first.subjects.columns)}")

    outcomes = first.options + ([topic.turnout[1]] if topic.turnout else [])
    predicted = {label: outcome_shares(run.normalized, run.topic) for label, run in runs.items()}
    answers = own_answers(first)
    for shares in predicted.values():
        answers = answers[answers.index.isin(shares.index)]
    if len(answers) < 2 * folds or answers.nunique() < 2:
        raise NotApplicable(f"{len(answers)} subjects with a known answer in every run are too few to train on")
    predicted = {label: shares.loc[answers.index, outcomes] for label, shares in predicted.items()}

    levels = first.dataset.levels
    design = np.hstack([
        indicators(banded(first.subjects.loc[answers.index, column], levels.get(column)), levels[column].several if column in levels else None)
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
        topic=topic, outcomes=outcomes, answers=answers, predicted=predicted, trained=trained,
        losses=pd.DataFrame(scored, index=answers.index), predictors=predictors, style=first.style,
    )


def describe(ground: Ground) -> str:
    counts = ground.answers.value_counts().reindex(ground.outcomes, fill_value=0)
    return "\n".join([
        f"{ground.topic.name}: {len(ground.answers)} subjects with a known answer in every run",
        f"  runs       {', '.join(ground.predicted)}",
        f"  trained on {', '.join(ground.predictors)}",
        f"  answers    {', '.join(f'{outcome} {count}' for outcome, count in counts.items())}",
    ])
