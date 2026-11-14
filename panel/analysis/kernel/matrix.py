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
from typing import Dict, List, Mapping, Sequence, Union

import numpy as np
import pandas as pd

from panel.dataset import Levels
from panel.describe.template import load_template
from panel.run.record import DELIVERED
from panel.run.resolve import REPEAT, SUBJECT
from panel.topic import PlotStyle, Topic

from ..aggregate import outcome_rows
from ..comparison import RunSet
from ..saved import NotApplicable, SavedRun
from ..stats.design import indicators
from ..stats.replicates import halves

MISSING = "(missing)"
PROFILE = "profile"


@dataclass(frozen=True)
class Kernel:
    """
    Attributes:
        label: what the run is called beside others.
        run: the saved run it was read from.
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
        """The run's own population as a distribution over the profiles."""
        return self.weights / self.weights.sum()

    def profile(self, number: int) -> str:
        """One profile's values, `column=value` joined."""
        return ", ".join(f"{column}={value}" for column, value in self.factors.loc[number].items())

    def grouping(self, column: str) -> pd.Series:
        """A factor as the groups analysis reads it: a numeric one cut into its declared bands, the rest as they are."""
        values, declared = self.factors[column], self.levels.get(column)
        if declared and declared.bands and pd.api.types.is_numeric_dtype(values):
            labels, tops = list(declared.bands), list(declared.bands.values())
            return pd.cut(values, [float("-inf"), *tops], labels=labels).astype(str)
        return values.astype(str)

    def design(self, columns: Union[Sequence[str], None] = None) -> Dict[str, np.ndarray]:
        """Factor -> its indicator columns, one per level (bands and several answers as declared)."""
        out = {}
        for column in columns or self.factors.columns:
            declared = self.levels.get(column)
            values = self.factors[column].replace(MISSING, np.nan) if declared and declared.several else self.factors[column]
            out[column] = indicators(values, declared.several if declared else None, declared.bands if declared else None, MISSING)
        return out

    def level_designs(self, columns: Union[Sequence[str], None] = None) -> Dict[str, pd.DataFrame]:
        """Factor -> its design as a table: a row per level the profiles hold (the raw value), its indicator columns."""
        return {
            column: pd.DataFrame(block, index=pd.Index(self.factors[column].to_numpy(), name=column)).groupby(level=0, sort=True).first()
            for column, block in self.design(columns).items()
        }

    def players(self) -> Dict[str, np.ndarray]:
        """
        The factors as players of an attribution: name -> design columns; the columns the
        dataset declares `together` are one player under its name, each other factor its own.
        """
        design = self.design()
        out, done = {}, set()
        for name, group in self.run.dataset.together.items():
            held = [column for column in group if column in design]
            if held:
                out[name] = np.hstack([design[column] for column in held])
                done |= set(held)
        return {**{column: design[column] for column in design if column not in done}, **out}

    def subset(self, profiles: Sequence[int]) -> "Kernel":
        """The kernel on some of its profiles (given by position): their subjects and answers only; never designed."""
        chosen = self.matrix.columns[np.asarray(profiles)]
        calls = self.calls[self.calls[PROFILE].isin(chosen)].reset_index(drop=True)
        return assembled(self, calls, self.factors.loc[chosen], designed=False)

    def resampled(self, random: np.random.Generator) -> "Kernel":
        """The kernel with every subject's repeats drawn again, with replacement, from its own."""
        positions = self.calls.groupby(SUBJECT).indices.values()
        drawn = np.concatenate([random.choice(found, len(found)) for found in positions])
        return assembled(self, self.calls.iloc[drawn].reset_index(drop=True), self.factors, self.designed)


@dataclass(frozen=True)
class KernelSet:
    """Label -> kernel, all of one topic, in the order they were named."""
    topic: Topic
    kernels: Dict[str, Kernel]
    style: PlotStyle

    @property
    def labels(self) -> Dict[str, str]:
        """Outcome -> the name a figure shows."""
        first = next(iter(self.kernels.values())).run
        return {outcome: first.labels.get(outcome, outcome) for outcome in next(iter(self.kernels.values())).outcomes}


def factor_columns(run: SavedRun) -> tuple:
    """The columns the run's description reads, or a stored description's `reads`."""
    return load_template(run.dataset.description_path(run.spec.description)).columns


def kernel_of(run: SavedRun, label: str, factors: Union[Sequence[str], None] = None) -> Kernel:
    """
    Read one saved run as a kernel.

    Args:
        run: the run.
        label: what to call it.
        factors: the profile's columns; those the description reads by default.

    Raises:
        NotApplicable: if the description names no column, so there is no profile.
        ValueError: on a factor the dataset does not have, or a numeric factor with a gap.
    """
    columns = list(factors or factor_columns(run))
    if not columns:
        raise NotApplicable(f"'{run.stem}': its description reads no column, so its subjects have no profile")
    unknown = sorted(set(columns) - set(run.subjects.columns))
    if unknown:
        raise ValueError(f"'{run.dataset.name}' has no columns {unknown}")
    values = run.subjects[columns]
    gaps = [column for column in columns if pd.api.types.is_numeric_dtype(values[column]) and values[column].isna().any()]
    if gaps:
        raise ValueError(f"'{run.stem}': numeric factors with missing values: {gaps}")
    values = values.astype({column: object for column in columns if not pd.api.types.is_numeric_dtype(values[column])}).fillna(MISSING)

    frame = run.normalized.reset_index()
    if REPEAT not in frame.columns:
        frame[REPEAT] = 1
    answers = outcome_rows(frame[run.topic.columns], run.topic)
    answers.insert(0, REPEAT, frame.loc[answers.index, REPEAT].to_numpy())
    answers.insert(0, SUBJECT, frame.loc[answers.index, SUBJECT].to_numpy())

    keys = values.loc[answers[SUBJECT].unique()]
    profile = keys.groupby(columns, sort=True, dropna=False).ngroup().rename(PROFILE)
    answers.insert(0, PROFILE, answers[SUBJECT].map(profile).to_numpy())
    calls = answers.reset_index(drop=True)
    profiles = keys.assign(**{PROFILE: profile}).drop_duplicates(PROFILE).set_index(PROFILE).sort_index()[columns]
    sizes = [profiles[column].nunique() for column in columns]
    designed = len(profiles) == int(np.prod(sizes)) and len(profiles) > 1
    delivered = run.record.get(DELIVERED) or {}
    empty = Kernel(label=label, run=run, matrix=pd.DataFrame(), factors=profiles, weights=pd.Series(dtype=float), calls=calls,
                   designed=designed, averaged=delivered.get("averaged"))
    return assembled(empty, calls, profiles, designed)


def assembled(kernel: Kernel, calls: pd.DataFrame, factors: pd.DataFrame, designed: bool) -> Kernel:
    """A kernel like `kernel`, its matrix and weights made from these answers."""
    outcomes = [column for column in calls.columns if column not in (PROFILE, SUBJECT, REPEAT)]
    means = calls.groupby(PROFILE)[outcomes].mean()
    means = means.div(means.sum(axis=1), axis=0)
    weights = calls.groupby(PROFILE)[SUBJECT].nunique().rename("subjects")
    return Kernel(label=kernel.label, run=kernel.run, matrix=means.T, factors=factors.loc[means.index], weights=weights, calls=calls,
                  designed=designed, averaged=kernel.averaged)


def load_kernels(runs: RunSet, factors: Union[Sequence[str], Mapping[str, Sequence[str]], None] = None) -> KernelSet:
    """
    Every run of a run set as a kernel.

    Args:
        runs: from `panel.analysis.comparison.load_runs`.
        factors: the profile's columns for every run, or label -> columns per run; the
            description's by default.
    """
    chosen = (lambda label: factors.get(label)) if isinstance(factors, Mapping) else (lambda label: factors)
    kernels = {label: kernel_of(run, label, chosen(label)) for label, run in runs.runs.items()}
    return KernelSet(topic=runs.topic, kernels=kernels, style=runs.style)


def push(kernel: Kernel, population: pd.Series) -> pd.Series:
    """
    What a population chooses by the kernel: `M q`.

    Args:
        kernel: the kernel.
        population: weights over the kernel's profiles, by profile; renormalised.

    Raises:
        ValueError: on a profile the kernel does not have, or weights that are negative or sum to zero.
    """
    unknown = population.index.difference(kernel.matrix.columns)
    if len(unknown):
        raise ValueError(f"'{kernel.label}' has no profiles {list(unknown[:5])}")
    if (population < 0).any() or population.sum() <= 0:
        raise ValueError("a population needs non-negative weights with a positive sum")
    weights = population.reindex(kernel.matrix.columns, fill_value=0.0)
    return kernel.matrix @ (weights / weights.sum())


def describe(kernels: KernelSet) -> str:
    """What was loaded, a line per kernel."""
    width = max(len(label) for label in kernels.kernels)
    lines = [f"{kernels.topic.name}: {len(kernels.kernels)} kernels, {len(next(iter(kernels.kernels.values())).outcomes)} outcomes"]
    for label, kernel in kernels.kernels.items():
        kind = "designed" if kernel.designed else "sampled"
        answers = f"means over {kernel.averaged} repeats" if kernel.averaged else f"{kernel.repeats} repeat{'s' if kernel.repeats > 1 else ''}"
        lines.append(f"  {label:<{width}}  {kernel.matrix.shape[1]} profiles of {int(kernel.weights.sum())} subjects, {kind}, {answers}; factors {', '.join(kernel.factors.columns)}")
    return "\n".join(lines)


HALVES, REPEATS = "half-samples of the profiles", "resampled repeats"


def scheme_of(kernel: Kernel) -> Union[str, None]:
    """
    Where a kernel's replicates come from: half-samples of the profiles for a sampled kernel,
    whose profiles are a draw from a population; resampled repeats for a designed one, whose
    cells are fixed and whose randomness is the model's; None for a designed kernel without
    repeats, or of delivered means.
    """
    if not kernel.designed:
        return HALVES
    return REPEATS if kernel.repeats > 1 else None


def replica_positions(kernel: Kernel, draws: int, seed: int = 0) -> List[np.ndarray]:
    """For half-sampling: the profiles' positions of each replicate; the same for every kernel of as many profiles, so replicates pair."""
    return halves(kernel.matrix.shape[1], draws, seed)


def replicas(kernel: Kernel, draws: int, seed: int = 0) -> List[Kernel]:
    """`draws` replicates of the kernel by its scheme (`scheme_of`); none when it has none."""
    scheme = scheme_of(kernel)
    if scheme == HALVES:
        return [kernel.subset(positions) for positions in replica_positions(kernel, draws, seed)]
    if scheme == REPEATS:
        random = np.random.default_rng(seed)
        return [kernel.resampled(random) for _ in range(draws)]
    return []
