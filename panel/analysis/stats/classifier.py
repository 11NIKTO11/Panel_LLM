"""
A model trained on the subjects' own answers, and how a set of predicted probabilities scores
against those answers.

The trained model is a ridge multinomial logit, and every subject's probabilities come from a
fit that did not see that subject (cross-validation), so they are held to the same standard
as a model that never saw any of the answers.
"""
from typing import Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

FLOOR = 0.01
SHIFT = 0.01


def cross_validated(features: np.ndarray, answers: np.ndarray, outcomes: Sequence[str], strength: float, folds: int = 5, seed: int = 0) -> np.ndarray:
    """
    Subjects by outcomes, in the order of `outcomes`; `strength` is the logit's `C`, the inverse of its
    ridge penalty. An outcome no subject of a fit answered gets zero there; a fit left with a single
    outcome raises in sklearn.
    """
    outcomes = list(outcomes)
    found = np.zeros((len(answers), len(outcomes)))
    for train, test in KFold(folds, shuffle=True, random_state=seed).split(features):
        model = LogisticRegression(C=strength, max_iter=5000).fit(features[train], answers[train])
        found[np.ix_(test, [outcomes.index(outcome) for outcome in model.classes_])] = model.predict_proba(features[test])
    return found


def recalibrated(probabilities: np.ndarray, answers: np.ndarray, outcomes: Sequence[str], folds: int = 5, seed: int = 0, shift: float = SHIFT) -> np.ndarray:
    """What the probabilities know about the answers, whatever their calibration; `shift` gives a zero a logarithm."""
    return cross_validated(np.log(np.asarray(probabilities, dtype=float) + shift), answers, outcomes, strength=1.0, folds=folds, seed=seed)


def losses(probabilities: np.ndarray, answered: np.ndarray, floor: float = FLOOR) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per subject, the log-loss (nats) and the Brier score; `answered` is 1 at each subject's outcome, 0 elsewhere.
    `floor` is the share of a uniform distribution mixed in first, so a zero on that outcome costs a finite amount.
    """
    mixed = (1 - floor) * np.asarray(probabilities, dtype=float) + floor / answered.shape[1]
    return -np.log((mixed * answered).sum(axis=1)), ((mixed - answered) ** 2).sum(axis=1)
