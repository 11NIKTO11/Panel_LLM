"""
A run read as a kernel: a matrix with one row per outcome and one column per distinct
profile, each column the answer the model gives that profile.

A profile is what the model was told about a subject: the values of the columns the run's
description reads (a stored description lists them as `reads`). Subjects with the same values
are one profile, and its column is the mean of their answers, repeats included. A population
is a distribution `q` over the profiles; the kernel pushes it forward to `M q`, what the
model says that population chooses.

A kernel is `designed` when its profiles are every combination of the factors' levels - a
grid - and `sampled` otherwise; it is read off the profiles, not declared.
"""
from dataclasses import dataclass
from typing import Dict, Sequence, Union

import numpy as np
import pandas as pd

from panel.dataset import Levels
from panel.run.resolve import REPEAT, SUBJECT
from panel.topic import PlotStyle, Topic

from ..levels import banded
from ..saved import SavedRun

MISSING = "(missing)"
PROFILE = "profile"


@dataclass(frozen=True)
class Kernel:
    """
    Attributes:
        matrix: outcomes x profiles, each column summing to 1.
        factors: profiles x factor columns, the values as the dataset writes them, a missing
            one as `(missing)`.
        weights: subjects per profile - the run's own population, unnormalised.
        calls: one row per answer: `profile`, `id`, `repeat`, then the outcomes.
        designed: every combination of the factors' levels is a profile, once.
        averaged: for a delivered run whose rows are means, over how many repeats; else None.
    """
    label: str
    run: SavedRun
    matrix: pd.DataFrame
    factors: pd.DataFrame
    weights: pd.Series
    calls: pd.DataFrame
    designed: bool
    averaged: Union[int, None] = None

    @property
    def outcomes(self) -> list:
        return list(self.matrix.index)

    @property
    def levels(self) -> Dict[str, Levels]:
        """The dataset's declarations of the factors that have one."""
        return {column: levels for column, levels in self.run.dataset.levels.items() if column in self.factors.columns}

    @property
    def repeats(self) -> int:
        """Answers per subject, the most any subject has."""
        return int(self.calls.groupby(SUBJECT).size().max())

    @property
    def population(self) -> pd.Series:
        return self.weights / self.weights.sum()

    def profile(self, number: int) -> str:
        return ", ".join(f"{column}={value}" for column, value in self.factors.loc[number].items())

    def grouping(self, column: str) -> pd.Series:
        """A factor as the groups analysis reads it: a numeric one cut into its declared bands, the rest as they are."""
        return banded(self.factors[column], self.levels.get(column)).astype(str)

    def subset(self, profiles: Sequence[int]) -> "Kernel":
        """The kernel on some of its profiles (given by position): their subjects and answers only; never designed."""
        chosen = self.matrix.columns[np.asarray(profiles)]
        calls = self.calls[self.calls[PROFILE].isin(chosen)].reset_index(drop=True)
        return assembled(self.label, self.run, calls, self.factors.loc[chosen], False, self.averaged)

    def resampled(self, random: np.random.Generator) -> "Kernel":
        """The kernel with every subject's repeats drawn again, with replacement, from its own."""
        positions = self.calls.groupby(SUBJECT).indices.values()
        drawn = np.concatenate([random.choice(found, len(found)) for found in positions])
        return assembled(self.label, self.run, self.calls.iloc[drawn].reset_index(drop=True), self.factors, self.designed, self.averaged)


@dataclass(frozen=True)
class KernelSet:
    """Label -> kernel, all of one topic, in the order they were named."""
    topic: Topic
    kernels: Dict[str, Kernel]
    style: PlotStyle

    @property
    def labels(self) -> Dict[str, str]:
        """Outcome -> the name a figure shows."""
        first = next(iter(self.kernels.values()))
        return {outcome: first.run.labels.get(outcome, outcome) for outcome in first.outcomes}


def assembled(label: str, run: SavedRun, calls: pd.DataFrame, factors: pd.DataFrame, designed: bool, averaged: Union[int, None] = None) -> Kernel:
    """`calls` and `factors` as `Kernel` holds them; profiles of `factors` without an answer are left out."""
    outcomes = [column for column in calls.columns if column not in (PROFILE, SUBJECT, REPEAT)]
    means = calls.groupby(PROFILE)[outcomes].mean()
    means = means.div(means.sum(axis=1), axis=0)
    weights = calls.groupby(PROFILE)[SUBJECT].nunique().rename("subjects")
    return Kernel(label=label, run=run, matrix=means.T, factors=factors.loc[means.index], weights=weights, calls=calls, designed=designed, averaged=averaged)


def push(kernel: Kernel, population: pd.Series) -> pd.Series:
    """`M q`; `population` is weights by profile, any sum, and need not name every profile."""
    unknown = population.index.difference(kernel.matrix.columns)
    if len(unknown):
        raise ValueError(f"'{kernel.label}' has no profiles {list(unknown[:5])}")
    if (population < 0).any() or population.sum() <= 0:
        raise ValueError("a population needs non-negative weights with a positive sum")
    weights = population.reindex(kernel.matrix.columns, fill_value=0.0)
    return kernel.matrix @ (weights / weights.sum())
