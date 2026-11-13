"""Matrices as heatmaps: a signed square one between the same things, and a grid of shares, rows by columns."""
from typing import Mapping, Union

import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure

from .style import INK, MUTED, SURFACE, legible, wrap


def heatmap(
    matrix: pd.DataFrame,
    bound: Union[float, None] = None,
    diagonal: bool = False,
    decimals: int = 2,
    labels: Union[Mapping[str, str], None] = None,
    title: Union[str, None] = None,
    note: Union[str, None] = None,
) -> Figure:
    """
    A signed square matrix, blue above zero and red below, every cell with its number.

    Args:
        matrix: indexed and columned by the same keys.
        bound: the value at full colour, symmetric around zero; the largest drawn magnitude
            by default.
        diagonal: draw the diagonal too. Off by default: a thing against itself says
            nothing and, in a covariance matrix, takes the whole colour range.
        decimals: of the numbers in the cells.
        labels: key -> display name; the key itself by default.
        title: figure title.
        note: a line under the title.
    """
    labels = labels or {}
    values = matrix.to_numpy(dtype=float).copy()
    if not diagonal:
        np.fill_diagonal(values, np.nan)
    bound = bound or float(np.nanmax(np.abs(values))) or 1.0
    size = 0.62 * len(matrix)
    height = size + 2.4
    fig = Figure(figsize=(size + 3.2, height), facecolor=SURFACE)
    ax = fig.subplots()
    ax.set_facecolor(SURFACE)
    image = ax.imshow(values, cmap="RdBu", norm=TwoSlopeNorm(0, -bound, bound))
    for (row, column), value in np.ndenumerate(values):
        if not np.isnan(value):
            ax.text(column, row, f"{round(value, decimals) + 0.0:.{decimals}f}", ha="center", va="center", fontsize=8 + 4.5 / decimals, color=SURFACE if abs(value) > 0.6 * bound else INK)
    names = wrap((labels.get(key, key) for key in matrix.index), 18)
    ax.set_xticks(range(len(matrix)), [str(labels.get(key, key)) for key in matrix.index], rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(matrix)), names)
    ax.set_xticks(np.arange(len(matrix) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(matrix) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0, colors=INK, labelsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, colors=MUTED, labelsize=10)
    if title:
        fig.suptitle(title, x=0.02, ha="left", fontsize=13, color=INK, fontweight="bold")
    if note:
        fig.text(0.02, 1 - 0.42 / height, note, ha="left", va="top", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 1 - (0.5 if note else 0.3) / height))
    return fig


def grid(
    table: pd.DataFrame,
    below: Union[pd.DataFrame, None] = None,
    decimals: int = 2,
    scale: float = 1.0,
    cmap: str = "Blues",
    signed: bool = False,
    title: Union[str, None] = None,
    note: Union[str, None] = None,
) -> Figure:
    """
    A rows-by-columns table of numbers, coloured by size, every cell with its number.

    Args:
        table: indexed by row, a column per group.
        below: the same shape, a text per cell written under its number (e.g. a p-value).
        decimals: of the numbers in the cells.
        scale: what the numbers are multiplied by before they are written (100 for shares in percent).
        cmap: the matplotlib colour range, from zero to the largest number.
        signed: the numbers may be negative: coloured by how far from zero, written with their sign.
        title: figure title.
        note: a line under the title.
    """
    values = table.to_numpy(dtype=float) * scale
    top = float(np.nanmax(np.abs(values))) or 1.0
    widest = max(len(str(column)) for column in table.columns)
    height = (0.6 if below is not None else 0.42) * len(table) + 1.9 + 0.07 * widest
    fig = Figure(figsize=(1.5 * table.shape[1] + 3.2, height), facecolor=SURFACE)
    ax = fig.subplots()
    ax.set_facecolor(SURFACE)
    image = ax.imshow(np.abs(values), cmap=cmap, vmin=0, vmax=top, aspect="auto")
    for (row, column), value in np.ndenumerate(values):
        if not np.isnan(value):
            ax.text(column, row, f"{value:{'+' if signed else ''}.{decimals}f}" + (f"\n{below.iat[row, column]}" if below is not None else ""), ha="center", va="center", fontsize=10, color=legible(image.cmap(image.norm(abs(value)))))
    ax.set_xticks(range(table.shape[1]), [str(column) for column in table.columns], rotation=30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(table)), [str(row) for row in table.index])
    ax.set_xticks(np.arange(table.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(table) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0, colors=INK, labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        fig.suptitle(title, x=0.02, ha="left", fontsize=13, color=INK, fontweight="bold")
    if note:
        fig.text(0.02, 1 - 0.42 / height, note, ha="left", va="top", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 1 - (0.5 if note else 0.3) / height))
    return fig


def grids(
    tables: Mapping[str, pd.DataFrame],
    decimals: int = 2,
    scale: float = 1.0,
    cmap: str = "Blues",
    title: Union[str, None] = None,
    note: Union[str, None] = None,
) -> Figure:
    """
    Several tables of non-negative numbers with the same rows and columns, one under the other
    on one colour range, every cell with its number.

    Args:
        tables: panel name -> its table, indexed by row, a column per group.
        decimals: of the numbers in the cells.
        scale: what the numbers are multiplied by before they are written.
        cmap: the matplotlib colour range, from zero to the largest number of all tables.
        title: figure title.
        note: a line under the title.
    """
    first = next(iter(tables.values()))
    values = {name: table.to_numpy(dtype=float) * scale for name, table in tables.items()}
    top = max((float(np.nanmax(found)) for found in values.values() if not np.isnan(found).all()), default=1.0) or 1.0
    widest = max(len(str(column)) for column in first.columns)
    height = len(tables) * (0.36 * len(first) + 0.45) + 1.4 + 0.07 * widest
    fig = Figure(figsize=(0.8 * first.shape[1] + 2.6, height), facecolor=SURFACE)
    axes = fig.subplots(len(tables), 1, sharex=True, squeeze=False).ravel()
    for ax, (name, found) in zip(axes, values.items()):
        ax.set_facecolor(SURFACE)
        image = ax.imshow(found, cmap=cmap, vmin=0, vmax=top, aspect="auto")
        for (row, column), value in np.ndenumerate(found):
            if not np.isnan(value):
                ax.text(column, row, f"{value:.{decimals}f}", ha="center", va="center", fontsize=8, color=legible(image.cmap(image.norm(value))))
        ax.set_title(name, fontsize=10, color=INK, loc="left", fontweight="bold")
        ax.set_yticks(range(len(first)), [str(row) for row in first.index])
        ax.set_xticks(np.arange(first.shape[1] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(first) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="both", length=0, colors=INK, labelsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[-1].set_xticks(range(first.shape[1]), [str(column) for column in first.columns], rotation=30, ha="right", rotation_mode="anchor")
    if title:
        fig.suptitle(title, x=0.02, ha="left", fontsize=13, color=INK, fontweight="bold")
    if note:
        fig.text(0.02, 1 - 0.42 / height, note, ha="left", va="top", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 1 - (0.5 if note else 0.3) / height))
    return fig
