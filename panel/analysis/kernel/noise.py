"""
The noise floor: how far apart two answers to the same profile are, against two random subjects.

Every distance between profiles is read against it - a factor whose swap moves the answer less
than the model's own noise does nothing. Two answers to the same profile come from:

- repeats: the same subject asked again (all pairs of its repeats);
- copies: different subjects with the same profile, each with its mean answer - for a sampled
  run without repeats, where a few subjects share every value.

A run of neither kind, and a delivered table of means, has no noise floor; the table says so.
Per outcome, the noise variance is half the mean squared difference of a pair: two independent
draws differ by twice the variance.
"""
import itertools
from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from panel.run.resolve import SUBJECT

from ..output import Output
from ..plots.distributions import overlaid
from ..saved import NotApplicable
from ..stats.distributions import tv
from .matrix import PROFILE, Kernel, KernelSet

NAME = "kernel_noise"
REPEATS, COPIES = "repeats", "copies"
FLOOR, RANDOM = "noise floor (mean TV)", "random pairs (mean TV)"


@dataclass(frozen=True)
class Noise:
    """
    Attributes:
        kind: `repeats`, `copies`, or None when there is nothing to measure it on.
        pairs: TV of every pair of answers to one profile.
        variance: per outcome, the noise variance; NaN without pairs.
        random: TV between random pairs of subjects' answers.
    """
    kind: Union[str, None]
    pairs: np.ndarray
    variance: pd.Series
    random: np.ndarray

    @property
    def floor(self) -> float:
        return float(self.pairs.mean()) if len(self.pairs) else float("nan")


def _differences(groups) -> np.ndarray:
    found = [first - second for rows in groups for first, second in itertools.combinations(rows, 2)]
    return np.array(found) if found else np.empty((0, 0))


def noise_floor(kernel: Kernel, draws: int = 20000, seed: int = 0) -> Noise:
    calls, outcomes = kernel.calls, kernel.outcomes
    subjects = calls.groupby([PROFILE, SUBJECT])[outcomes].mean()
    if kernel.repeats > 1:
        kind, groups = REPEATS, [frame.to_numpy() for _, frame in calls.groupby(SUBJECT)[outcomes]]
    elif (kernel.weights > 1).any():
        kind, groups = COPIES, [frame.to_numpy() for _, frame in subjects.groupby(level=PROFILE) if len(frame) > 1]
    else:
        kind, groups = None, []
    differences = _differences(groups)
    random = np.random.default_rng(seed).integers(0, len(subjects), (2, draws))
    values = subjects.to_numpy()
    return Noise(
        kind=kind,
        pairs=0.5 * np.abs(differences).sum(axis=1) if len(differences) else np.empty(0),
        variance=pd.Series((differences ** 2).mean(axis=0) / 2 if len(differences) else np.nan, index=outcomes),
        random=tv(values[random[0]], values[random[1]]),
    )


def noise(kernels: KernelSet, out: Output, draws: int = 20000, seed: int = 0) -> Tuple[pd.DataFrame, List[Figure]]:
    found = {label: noise_floor(kernel, draws, seed) for label, kernel in kernels.kernels.items()}
    if not any(len(each.pairs) for each in found.values()):
        why = {label: f"means over {kernel.averaged} repeats" if kernel.averaged else "one answer per profile" for label, kernel in kernels.kernels.items()}
        raise NotApplicable(f"no kernel has two answers to one profile: {why}")
    rows = {}
    for label, each in found.items():
        rows[label] = {
            "from": each.kind or "none", "pairs": len(each.pairs), "differing": int((each.pairs > 1e-9).sum()),
            FLOOR: each.floor, "largest": float(each.pairs.max()) if len(each.pairs) else np.nan,
            RANDOM: float(each.random.mean()), "floor / random": each.floor / float(each.random.mean()),
        }
    table = out.table(NAME, pd.DataFrame(rows).T.rename_axis("kernel"))
    panels = {label: {"random pairs": each.random, **({"same profile": each.pairs} if len(each.pairs) else {})} for label, each in found.items()}
    marks = {label: {"floor": each.floor} for label, each in found.items() if len(each.pairs)}
    figure = overlaid(
        panels, marks, xlabel="TV between two answers",
        title=f"{kernels.topic.name}: two answers to the same profile against two random subjects",
        note="same profile: repeats of one subject, or subjects sharing every value; the line is the noise floor, their mean",
    )
    return table, [out.figure(NAME, figure)]
