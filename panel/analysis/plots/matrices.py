"""Heatmaps of matrices: covariance and correlation, per-group error grids, spread by group."""
from math import ceil
from typing import Dict, Mapping, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .style import wrap


def heatmap(
    matrix: pd.DataFrame,
    title: Union[str, None] = None,
    vmin: Union[float, None] = None,
    vmax: Union[float, None] = None,
    fmt: str = ".4f",
    cmap: str = "coolwarm_r",
    labels: Union[Mapping[str, str], None] = None,
    wrap_width: int = 21,
) -> plt.Figure:
    """
    A square matrix as an annotated heatmap, symmetric colour range around zero unless given.

    Args:
        matrix: indexed and columned by the same keys, e.g. a covariance or correlation matrix.
        labels: key -> display name for the ticks.
    """
    labels = labels or {}
    if vmin is None and vmax is None:
        bound = float(matrix.abs().max().max())
        vmin, vmax = -bound, bound
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, ax=ax, annot=True, cmap=cmap, fmt=fmt, linewidths=0.5, vmin=vmin, vmax=vmax)
    names = wrap((labels.get(key, key) for key in matrix.index), wrap_width)
    ax.set_yticklabels(names, rotation=0, ha="right", va="center")
    ax.set_xticklabels(names, rotation=90, ha="right", va="center", rotation_mode="anchor")
    if title:
        ax.set_title(title, fontsize=18)
    fig.tight_layout()
    return fig


def error_grid(
    table: pd.DataFrame,
    sources: Sequence[str],
    metric: str = "MAE",
    ncols: int = 4,
    vmin: float = 0,
    vmax: float = 5,
    figsize_per_plot: Tuple[float, float] = (4, 3.5),
    title: Union[str, None] = None,
) -> plt.Figure:
    """
    One small symmetric heatmap per group of pairwise errors between sources.

    Args:
        table: indexed by group, one column `A vs B` per pair of `sources`, as `compare.pairwise` writes.
        sources: the sources, in the order they appear on both axes of each small matrix.
        metric: name for the colour bar; values at most 1 are shown as percentages.
    """
    nrows = ceil(len(table) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows), squeeze=False)
    axes = axes.flatten()
    scale = 100 if table.select_dtypes("number").max().max() <= 1.0 else 1
    for ax, (group, row) in zip(axes, table.iterrows()):
        errors = np.zeros((len(sources), len(sources)))
        for i, first in enumerate(sources):
            for j, second in enumerate(sources):
                if i != j:
                    column = f"{first} vs {second}" if f"{first} vs {second}" in row.index else f"{second} vs {first}"
                    errors[i, j] = row[column] * scale
        sns.heatmap(errors, ax=ax, annot=True, fmt=".1f", cmap="plasma", vmin=vmin, vmax=vmax,
                    xticklabels=list(sources), yticklabels=list(sources), cbar_kws={"label": f"{metric}" + (" (%)" if scale == 100 else "")})
        ax.set_title(str(group), fontsize=10, fontweight="bold")
    for ax in axes[len(table):]:
        ax.axis("off")
    fig.suptitle(title or f"{metric} between sources, by group", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def spread_heatmap(
    matrix: pd.DataFrame,
    title: Union[str, None] = None,
    cmap: str = "viridis",
    fmt: str = ".3f",
    labels: Union[Mapping[str, str], None] = None,
    width_per_column: float = 1.2,
    height_per_row: float = 0.6,
) -> plt.Figure:
    """A rows x groups matrix (e.g. standard deviation of each option per region) as one heatmap."""
    labels = labels or {}
    fig, ax = plt.subplots(figsize=(max(6, matrix.shape[1] * width_per_column), max(2, matrix.shape[0] * height_per_row)))
    sns.heatmap(matrix, ax=ax, mask=matrix.isna(), cmap=cmap, annot=True, fmt=fmt, linewidths=0.5, linecolor="white", cbar_kws={"label": "Std. dev."})
    ax.set_yticklabels([labels.get(key, key) for key in matrix.index], rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def spread_heatmap_grid(
    matrices: Dict[str, pd.DataFrame],
    ncols: int = 4,
    per_subplot_size: Tuple[float, float] = (4.0, 3.0),
    cmap: str = "viridis",
    annot: bool = False,
    fmt: str = ".3f",
    title: Union[str, None] = None,
) -> plt.Figure:
    """Several spread matrices, one small heatmap each on a shared colour scale."""
    names = list(matrices)
    cols = min(ncols, len(names))
    rows = max(1, ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * per_subplot_size[0], rows * per_subplot_size[1]), squeeze=False)
    stacked = pd.concat([matrix.stack() for matrix in matrices.values() if not matrix.isna().all().all()], axis=0)
    vmin, vmax = (float(stacked.min()), float(stacked.max())) if not stacked.empty else (0.0, 1.0)
    for index, name in enumerate(names):
        ax = axes[index // cols][index % cols]
        matrix = matrices[name]
        sns.heatmap(matrix, ax=ax, mask=matrix.isna(), cmap=cmap, annot=annot, fmt=fmt, cbar=False, vmin=vmin, vmax=vmax,
                    linewidths=0.3, linecolor="white", xticklabels=True, yticklabels=True)
        ax.set_title(str(name))
        ax.set_xlabel("")
        ax.set_ylabel("")
        if index % cols:
            ax.set_yticklabels([])
        if index // cols != rows - 1:
            ax.set_xticklabels([])
    for index in range(len(names), rows * cols):
        axes[index // cols][index % cols].axis("off")
    scalar = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    scalar.set_array([])
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.15, top=0.9, wspace=0.15, hspace=0.3)
    fig.colorbar(scalar, cax=fig.add_axes([0.92, 0.15, 0.02, 0.7]), label="Std. dev.")
    if title:
        fig.suptitle(title, fontsize=14)
    return fig
