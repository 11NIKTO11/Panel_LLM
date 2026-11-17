"""
How far the profiles carry the answers: per outcome over the profiles, per profile, and
between two profiles.

- per outcome: mean, standard deviation and range of its probability over the profiles;
- per profile: the entropy of its answer, its largest probability, its TV from the mean answer;
- between profiles: the TV of every pair. Its largest is the Dobrushin coefficient `δ`, which
  bounds how far a population's answer can move: `TV(Mq, Mq') ≤ δ TV(q, q')`. One odd profile
  sets `δ`, so the 99th percentile stands beside it as a robust largest.

Profiles count once each here, whatever their number of subjects.
"""
from typing import List, Tuple

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..output import Output
from ..plots.distributions import overlaid
from ..plots.matrices import grid
from ..saved import NotApplicable
from ..stats.distributions import pairwise_tv, tv
from .matrix import KernelSet
from .noise import noise_floor

NAME, OUTCOMES = "kernel_scales", "kernel_scales_outcomes"
DOBRUSHIN, ROBUST = "Dobrushin δ", "q99 pairwise TV"


def scales(kernels: KernelSet, out: Output) -> Tuple[pd.DataFrame, List[Figure]]:
    rows, outcomes, panels, marks = {}, {}, {}, {}
    for label, kernel in kernels.kernels.items():
        columns = kernel.matrix.to_numpy().T
        if len(columns) < 2:
            raise NotApplicable(f"'{label}' has a single profile")
        distances, (largest, first, second) = pairwise_tv(columns)
        from_mean = tv(columns, columns.mean(axis=0))
        quantiles = np.quantile(distances, [0.5, 0.9, 0.99])
        entropy = -(columns * np.log2(np.where(columns > 0, columns, 1))).sum(axis=1)
        floor = noise_floor(kernel).floor
        rows[label] = {
            DOBRUSHIN: largest, "q50 pairwise TV": quantiles[0], "q90 pairwise TV": quantiles[1], ROBUST: quantiles[2],
            "mean pairwise TV": float(distances.mean()), "mean TV from the mean answer": float(from_mean.mean()),
            "largest TV from the mean answer": float(from_mean.max()), "noise floor": floor,
            "mean entropy (bits)": float(entropy.mean()), "mean largest probability": float(columns.max(axis=1).mean()),
            "pair setting δ": f"{kernel.profile(kernel.matrix.columns[first])}  |  {kernel.profile(kernel.matrix.columns[second])}",
        }
        matrix = kernel.matrix
        outcomes[label] = pd.DataFrame({"mean": matrix.mean(axis=1), "std": matrix.std(axis=1, ddof=0), "min": matrix.min(axis=1), "max": matrix.max(axis=1)})
        panels[label] = {"pairs of profiles": distances}
        marks[label] = {"δ": largest, "q99": quantiles[2], **({"noise": floor} if not np.isnan(floor) else {})}
    table = out.table(NAME, pd.DataFrame(rows).T.rename_axis("kernel"))
    out.table(OUTCOMES, pd.concat(outcomes, names=["kernel", "outcome"]))
    spread = pd.DataFrame({label: frame["std"] for label, frame in outcomes.items()}).rename(index=kernels.labels)
    figures = [
        overlaid(panels, marks, xlabel="TV between the answers of two profiles", title=f"{kernels.topic.name}: how far apart two profiles' answers are",
                 note="every pair of profiles; δ is the largest, q99 its robust stand-in, noise the floor of two answers to one profile"),
        grid(spread, decimals=1, scale=100, title=f"{kernels.topic.name}: how much each outcome moves over the profiles",
             note="standard deviation of the outcome's probability over the profiles, percentage points"),
    ]
    return table, [out.figure(NAME, figures[0]), out.figure(OUTCOMES, figures[1])]
