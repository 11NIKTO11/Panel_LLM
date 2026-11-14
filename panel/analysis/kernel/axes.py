"""
The directions the profiles move the answer in: the SVD of the kernel with each outcome's mean
over the profiles taken off (`stats.spectral`).

- an axis is a pattern over the outcomes - which rise as which fall - and its share is how much
  of the variation over the profiles lies along it;
- the soft ranks count the axes that matter; 1 is a single direction (turnout alone, say);
- each axis's profile scores are explained by the factors with ε², which says what drives it.
"""
from typing import List, Tuple

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..output import Output
from ..plots.curves import curves
from ..plots.matrices import grid, images
from ..stats.spectral import centred_svd, soft_ranks
from ..stats.variance import eps2
from .matrix import KernelSet

NAME, LOADINGS, DRIVERS, MATRIX = "kernel_axes", "kernel_axes_loadings", "kernel_axes_drivers", "kernel_axes_matrix"
SHOWN, DRIVEN = 5, 2
AXIS, SHARE = "axis", "share"


def axes(kernels: KernelSet, out: Output) -> Tuple[pd.DataFrame, List[Figure]]:
    """
    Per kernel the axes' shares and soft ranks; each axis's pattern; what drives the first two.

    Returns:
        The table - a row per kernel: the share of the first five axes and the three soft ranks -
        with the patterns (`kernel_axes_loadings`: pp per standard deviation along an axis, first
        three axes) and the drivers (`kernel_axes_drivers`: ε² of each factor for the scores on
        the first two) beside it, and the figures: the shares by axis, the patterns, the drivers,
        and the matrices as images with the profiles sorted by their score on axis 1.
    """
    rows, scree, loadings, drivers, sorted_matrices = {}, {}, {}, {}, {}
    for label, kernel in kernels.kernels.items():
        left, singular, right = centred_svd(kernel.matrix.to_numpy())
        shares = singular ** 2 / (singular ** 2).sum()
        rows[label] = {**{f"axis {k + 1} share": float(shares[k]) if k < len(shares) else np.nan for k in range(SHOWN)}, **soft_ranks(singular)}
        scree[label] = pd.DataFrame({AXIS: np.arange(1, len(shares)), SHARE: 100 * shares[:-1]})
        count = right.shape[1]
        loadings[label] = pd.DataFrame(100 * left[:, :3] * singular[:3] / np.sqrt(count), index=kernel.outcomes,
                                       columns=[f"axis {k + 1} ({100 * shares[k]:.0f} %)" for k in range(min(3, len(shares)))])
        drivers[label] = pd.DataFrame({f"axis {k + 1}": {column: eps2(right[k], kernel.grouping(column).to_numpy())[0] for column in kernel.factors.columns}
                                       for k in range(min(DRIVEN, len(right)))})
        sorted_matrices[label] = kernel.matrix.iloc[:, np.argsort(right[0])].rename(index=kernels.labels)
    table = out.table(NAME, pd.DataFrame(rows).T.rename_axis("kernel"))
    out.table(LOADINGS, pd.concat(loadings, names=["kernel", "outcome"]))
    out.table(DRIVERS, pd.concat(drivers, names=["kernel", "factor"]))
    figures = [out.figure(NAME, curves(scree, AXIS, SHARE, xlabel="axis, largest first", ylabel="% of the variation over the profiles",
                                       title=f"{kernels.topic.name}: how many directions the profiles move the answer in",
                                       note="share of each axis of the centred kernel; the last, always zero, is left out"))]
    for number, (label, frame) in enumerate(loadings.items(), start=1):
        figures.append(out.figure(f"{LOADINGS}_{number}", grid(frame.rename(index=kernels.labels), decimals=1, signed=True, cmap="Oranges",
                                                            title=f"{label}: the patterns of the first three axes",
                                                            note="change of each outcome's probability, pp, for a profile one standard deviation along the axis")))
    together = pd.concat({label: frame for label, frame in drivers.items()}, axis=1)
    together.columns = [f"{label}\n{axis}" for label, axis in together.columns]
    figures.append(out.figure(DRIVERS, grid(together.clip(lower=0), decimals=2, cmap="Greens", title=f"{kernels.topic.name}: what drives each axis",
                                            note="ε² of each factor for the profiles' scores on the axis; empty where a kernel has no such factor")))
    figures.append(out.figure(MATRIX, images(sorted_matrices, xlabel="profiles, sorted by their score on axis 1",
                                             title=f"{kernels.topic.name}: the kernels, a pixel column per profile")))
    return table, figures
