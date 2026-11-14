"""
A logit-additive surrogate of a kernel: `f(x) = softmax(b + Σ_factors E_factor[x_factor])`, one
vector of effects over the outcomes per level of each factor, fitted to the observed profiles'
answers by soft-label cross-entropy (a multinomial logit weighted by the answers' probabilities).

What it is for:

- additivity: how well a model without interactions reproduces the answers, out of sample;
- extrapolation: an answer for a combination of levels no subject had;
- the full product: the largest TV between any two combinations, found by coordinate ascent -
  every value it reports is reached by an explicit pair of combinations - and each outcome's
  bounds over the whole product, analytic (the sum of each factor's extremes) and searched.

A factor's design is a table with one row per level (its raw value) and its indicator columns,
so bands and several answers per cell enter as the caller built them. Two outcomes fit a binary
logit, written as the same softmax with the first outcome's effects at zero.
"""
from typing import Callable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .distributions import tv


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return scores / scores.sum(axis=-1, keepdims=True)


class LogitAdditive:
    """
    Attributes:
        designs: factor -> its design, a row per level.
        intercept: one per outcome.
        effects: per factor, levels x outcomes, in the order of `designs`.
    """

    def __init__(self, designs: Mapping[str, pd.DataFrame], factors: pd.DataFrame, answers: np.ndarray, strength: float = 1.0):
        """
        Fit the surrogate.

        Args:
            designs: factor -> a row per level (indexed by the raw value), its indicator columns.
            factors: one row per observed profile, a column per factor of `designs`.
            answers: the same profiles' answers, profiles x outcomes, rows summing to 1.
            strength: the logit's `C`, the inverse of its ridge penalty.

        Raises:
            ValueError: on a level the designs do not hold, or fewer than two outcomes.
        """
        self.designs = dict(designs)
        answers = np.asarray(answers, dtype=float)
        count = answers.shape[1]
        if count < 2:
            raise ValueError("a surrogate needs two outcomes or more")
        features = self.features(factors)
        weights = answers.ravel()
        kept = weights > 1e-12
        fitted = LogisticRegression(C=strength, max_iter=5000).fit(
            np.repeat(features, count, axis=0)[kept], np.tile(np.arange(count), len(features))[kept], sample_weight=weights[kept])
        coefficients, intercept = fitted.coef_.T, fitted.intercept_
        if coefficients.shape[1] == 1:
            coefficients = np.column_stack([np.zeros(len(coefficients)), coefficients[:, 0]])
            intercept = np.array([0.0, intercept[0]])
        offsets = np.cumsum([0] + [design.shape[1] for design in self.designs.values()])
        self.intercept = intercept
        self.effects = [design.to_numpy(dtype=float) @ coefficients[offsets[k]:offsets[k + 1]] for k, design in enumerate(self.designs.values())]

    def features(self, factors: pd.DataFrame) -> np.ndarray:
        """The design rows of each profile's levels, side by side."""
        unknown = {column: sorted(set(factors[column]) - set(design.index), key=str)[:5] for column, design in self.designs.items() if not set(factors[column]) <= set(design.index)}
        if unknown:
            raise ValueError(f"levels the surrogate has no design for: {unknown}")
        return np.hstack([design.loc[factors[column]].to_numpy(dtype=float) for column, design in self.designs.items()])

    def codes(self, factors: pd.DataFrame) -> np.ndarray:
        """Each profile's level positions, profiles x factors."""
        return np.column_stack([design.index.get_indexer(factors[column]) for column, design in self.designs.items()])

    def levels(self, codes: Sequence[int]) -> List:
        """The raw values at these level positions, one per factor."""
        return [design.index[code] for design, code in zip(self.designs.values(), codes)]

    def score(self, codes: np.ndarray) -> np.ndarray:
        return self.intercept + sum(effect[codes[k]] for k, effect in enumerate(self.effects))

    def predict(self, factors: pd.DataFrame) -> np.ndarray:
        """Answers for these profiles (observed or not), profiles x outcomes."""
        return softmax(self.score(self.codes(factors).T))

    def search(self, objective: Callable[[List[np.ndarray]], np.ndarray], starts: Sequence[Sequence[np.ndarray]]) -> List[Tuple[float, List[np.ndarray]]]:
        """
        Coordinate ascent over the full product from each start (a tuple of code vectors, one per
        combination the objective takes); every value found is reached by its combinations.
        """
        found = []
        for start in starts:
            combinations = [np.array(codes, copy=True) for codes in start]
            best = float(objective([self.score(codes) for codes in combinations]))
            better = True
            while better:
                better = False
                for k, effect in enumerate(self.effects):
                    for which in range(len(combinations)):
                        scores = [self.score(codes) for codes in combinations]
                        scores[which] = scores[which] - effect[combinations[which][k]] + effect
                        values = objective(scores)
                        at = int(np.argmax(values))
                        if values[at] > best + 1e-12:
                            best, better = float(values[at]), True
                            combinations[which][k] = at
            found.append((best, combinations))
        return found

    def largest_tv(self, starts: Sequence[Sequence[np.ndarray]]) -> List[Tuple[float, List[np.ndarray]]]:
        """The search for the pair of combinations whose answers are furthest apart."""
        return self.search(lambda scores: tv(softmax(scores[0]), softmax(scores[1])), starts)

    def extreme(self, outcome: int, sign: float, starts: Sequence[Sequence[np.ndarray]]) -> List[Tuple[float, List[np.ndarray]]]:
        """The search for the combination with the highest (`sign` 1) or lowest (-1) probability of one outcome."""
        return self.search(lambda scores: sign * softmax(scores[0])[..., outcome], starts)

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Per outcome, analytic lower and upper bounds of its probability over the full product."""
        count = len(self.intercept)
        low, high = np.zeros(count), np.zeros(count)
        for outcome in range(count):
            shifted = [effect - effect[:, [outcome]] for effect in self.effects]
            least = self.intercept - self.intercept[outcome] + sum(part.min(axis=0) for part in shifted)
            most = self.intercept - self.intercept[outcome] + sum(part.max(axis=0) for part in shifted)
            high[outcome], low[outcome] = 1 / np.exp(least).sum(), 1 / np.exp(most).sum()
        return low, high
