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
    Each subject's probabilities over the outcomes, from a logit fitted without that subject's fold.

    Args:
        features: subjects by design columns.
        answers: each subject's outcome.
        outcomes: every outcome, in the order of the columns returned.
        strength: the logit's `C`: the inverse of its ridge penalty.
        folds: how many parts the subjects are split into.
        seed: of the split.

    Returns:
        Subjects by outcomes; an outcome no subject of a fit answered gets zero there.

    Raises:
        ValueError: if a fit is left with a single outcome.
    """
    outcomes = list(outcomes)
    found = np.zeros((len(answers), len(outcomes)))
    for train, test in KFold(folds, shuffle=True, random_state=seed).split(features):
        model = LogisticRegression(C=strength, max_iter=5000).fit(features[train], answers[train])
        found[np.ix_(test, [outcomes.index(outcome) for outcome in model.classes_])] = model.predict_proba(features[test])
    return found


def recalibrated(probabilities: np.ndarray, answers: np.ndarray, outcomes: Sequence[str], folds: int = 5, seed: int = 0, shift: float = SHIFT) -> np.ndarray:
    """
    The probabilities read through a cross-validated logit on their logarithms: what they
    know about the answers, whatever their calibration.

    Args:
        probabilities: subjects by outcomes.
        answers: each subject's outcome.
        outcomes: every outcome, in column order.
        folds: of the cross-validation.
        seed: of the split.
        shift: added before the logarithm, so a zero has one.
    """
    return cross_validated(np.log(np.asarray(probabilities, dtype=float) + shift), answers, outcomes, strength=1.0, folds=folds, seed=seed)


def losses(probabilities: np.ndarray, answered: np.ndarray, floor: float = FLOOR) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per subject, the log-loss (nats) and the Brier score of the probabilities.

    Args:
        probabilities: subjects by outcomes, rows summing to 1.
        answered: the same shape, 1 at each subject's outcome and 0 elsewhere.
        floor: the share of a uniform distribution mixed in first, so a zero on the
            answered outcome costs a finite amount.
    """
    mixed = (1 - floor) * np.asarray(probabilities, dtype=float) + floor / answered.shape[1]
    return -np.log((mixed * answered).sum(axis=1)), ((mixed - answered) ** 2).sum(axis=1)
