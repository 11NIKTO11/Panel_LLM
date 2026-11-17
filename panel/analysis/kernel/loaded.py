"""
Saved runs read as kernels: which columns make a profile, and the kernel of one run or of a set.
"""
from typing import Mapping, Sequence, Union

import numpy as np
import pandas as pd

from panel.describe.template import load_template
from panel.run.record import DELIVERED
from panel.run.resolve import REPEAT, SUBJECT

from ..aggregate import outcome_rows
from ..comparison import RunSet
from ..saved import NotApplicable, SavedRun
from .matrix import MISSING, PROFILE, Kernel, KernelSet, assembled


def factor_columns(run: SavedRun) -> tuple:
    """The columns the run's description reads, or a stored description's `reads`."""
    return load_template(run.dataset.description_path(run.spec.description)).columns


def kernel_of(run: SavedRun, label: str, factors: Union[Sequence[str], None] = None) -> Kernel:
    """`factors` are the profile's columns, those the description reads by default."""
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
    profiles = keys.assign(**{PROFILE: profile}).drop_duplicates(PROFILE).set_index(PROFILE).sort_index()[columns]
    sizes = [profiles[column].nunique() for column in columns]
    designed = len(profiles) == int(np.prod(sizes)) and len(profiles) > 1
    delivered = run.record.get(DELIVERED) or {}
    return assembled(label, run, answers.reset_index(drop=True), profiles, designed, delivered.get("averaged"))


def load_kernels(runs: RunSet, factors: Union[Sequence[str], Mapping[str, Sequence[str]], None] = None) -> KernelSet:
    """`factors` are the profile's columns for every run, or label -> columns per run; the description's by default."""
    chosen = (lambda label: factors.get(label)) if isinstance(factors, Mapping) else (lambda label: factors)
    kernels = {label: kernel_of(run, label, chosen(label)) for label, run in runs.runs.items()}
    return KernelSet(topic=runs.topic, kernels=kernels, style=runs.style)


def describe(kernels: KernelSet) -> str:
    width = max(len(label) for label in kernels.kernels)
    lines = [f"{kernels.topic.name}: {len(kernels.kernels)} kernels, {len(next(iter(kernels.kernels.values())).outcomes)} outcomes"]
    for label, kernel in kernels.kernels.items():
        kind = "designed" if kernel.designed else "sampled"
        answers = f"means over {kernel.averaged} repeats" if kernel.averaged else f"{kernel.repeats} repeat{'s' if kernel.repeats > 1 else ''}"
        lines.append(f"  {label:<{width}}  {kernel.matrix.shape[1]} profiles of {int(kernel.weights.sum())} subjects, {kind}, {answers}; factors {', '.join(kernel.factors.columns)}")
    return "\n".join(lines)
