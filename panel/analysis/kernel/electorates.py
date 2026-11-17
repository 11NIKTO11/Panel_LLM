"""
How alike the options' electorates are: an option's electorate is the run's population weighted
by the probability of choosing it, and two options are close when their electorates have the
same make-up, factor by factor.

Per factor, an electorate's make-up is its distribution over the factor's levels; the distance
of two options is the TV of their make-ups, averaged over the factors (`llm_pushforward`'s
`party_distances(kind='electorate')`). The other kind there, correlation over the profiles, is
`compass`'s already.
"""
from typing import List, Tuple

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..output import Output
from ..plots.matrices import grid
from ..plots.style import DISTANCES
from ..saved import NotApplicable
from ..stats.distributions import tv
from .matrix import KernelSet

NAME = "kernel_electorates"


def electorates(kernels: KernelSet, out: Output) -> Tuple[pd.DataFrame, List[Figure]]:
    options = list(kernels.topic.options)
    if len(options) < 3:
        raise NotApplicable(f"'{kernels.topic.name}' has {len(options)} options; a map of distances between them needs three")
    found = {}
    for label, kernel in kernels.kernels.items():
        weighted = kernel.matrix.loc[options].to_numpy() * kernel.weights.reindex(kernel.matrix.columns).to_numpy()
        distances = np.zeros((len(options), len(options)))
        for column in kernel.factors.columns:
            levels = pd.get_dummies(kernel.grouping(column)).to_numpy(dtype=float)
            makeup = weighted @ levels
            makeup = makeup / makeup.sum(axis=1, keepdims=True)
            distances += tv(makeup[:, None], makeup[None]) / kernel.factors.shape[1]
        found[label] = pd.DataFrame(distances, index=options, columns=options)
    table = out.table(NAME, pd.concat(found, names=["kernel", "option"]))
    figures = [
        out.figure(f"{NAME}_{number}", grid(frame.rename(index=kernels.labels, columns=kernels.labels), decimals=2, cmap=DISTANCES,
                                           title=f"{label}: how alike the options' electorates are",
                                           note="TV between two options' electorates, factor by factor, averaged over the factors; 0 = the same make-up"))
        for number, (label, frame) in enumerate(found.items(), start=1)
    ]
    return table, figures
