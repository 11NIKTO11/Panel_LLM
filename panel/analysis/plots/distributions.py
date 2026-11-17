"""How values are spread: a stem per value of a column, histograms side by side, distributions overlaid per panel, and boxes with their points."""
from typing import Mapping, Sequence, Union

import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .style import EDGE, INK, MUTED, NEUTRAL, PANEL, SERIES, SURFACE, clean, finish, framed, tint, wrap


def spikes(
    shares: Mapping[str, pd.Series],
    rows: Sequence[str],
    head: Sequence[str] = (),
    colors: Union[Mapping[str, str], None] = None,
    labels: Union[Mapping[str, str], None] = None,
    title: Union[str, None] = None,
    note: Union[str, None] = None,
) -> Figure:
    """
    One row per column, a stem at every value it holds - a hairline ending in a dot, since a
    value is a point and not a range - as tall as the share of cells holding exactly it; the
    tallest of a row is named with its share. Stems of one panel share a scale.

    - `shares`: column -> share per value, the values between 0 and 1.
    - `head`: columns drawn in a panel of their own above the main one (e.g. turnout).
    """
    colors, labels = colors or {}, labels or {}
    panels = [list(part) for part in (head, rows) if len(part)]
    height = 1.5 + 0.62 * sum(len(part) for part in panels) + 0.4 * len(panels)
    fig = Figure(figsize=(9, height), facecolor=SURFACE)
    axes = fig.subplots(len(panels), 1, squeeze=False, gridspec_kw={"height_ratios": [len(part) for part in panels]})[:, 0]
    for ax, part in zip(axes, panels):
        clean(ax)
        tallest = max(float(shares[column].max()) for column in part)
        for position, column in enumerate(part):
            found = shares[column]
            base = position + 0.4
            ax.hlines(base, 0, 1, color=EDGE, linewidth=0.6)
            at, tops = found.index.to_numpy(dtype=float), base - 0.8 * found.to_numpy() / tallest
            ax.vlines(at, base, tops, color=colors.get(column, NEUTRAL), linewidth=0.9)
            ax.scatter(at, tops, s=13, color=colors.get(column, NEUTRAL), linewidths=0, zorder=3)
            top = found.idxmax()
            ax.annotate(f"{top:g} in {found.max():.0%}", (float(top), base - 0.8 * found.max() / tallest), xytext=(6, 3), textcoords="offset points", va="top", fontsize=8, color=MUTED)
        ax.set_yticks(np.arange(len(part)), wrap(labels.get(column, column) for column in part))
        ax.set_ylim(len(part) - 0.4, -0.6)
        ax.set_xlim(-0.01, 1.01)
        ax.xaxis.set_major_locator(mtick.MultipleLocator(0.1))
    return finish(fig, height, title, note)


def histograms(
    values: pd.DataFrame,
    ylabel: str,
    bins: int = 30,
    title: Union[str, None] = None,
    note: Union[str, None] = None,
) -> Figure:
    """One framed histogram per column, each on its own scale, the mean marked; missing values are left out."""
    height = 4.2
    fig = Figure(figsize=(4.4 * values.shape[1] + 0.6, height), facecolor=SURFACE)
    axes = fig.subplots(1, values.shape[1], squeeze=False)[0]
    for ax, column in zip(axes, values.columns):
        framed(ax)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color=SURFACE, linewidth=0.8)
        found = values[column].dropna().to_numpy(dtype=float)
        ax.hist(found, bins=bins, color=SERIES[0], edgecolor=PANEL, linewidth=0.6)
        ax.axvline(found.mean(), color=INK, linewidth=1)
        ax.annotate(f"mean {found.mean():.3f}", (found.mean(), 1), xycoords=("data", "axes fraction"), xytext=(4, -12), textcoords="offset points", fontsize=8, color=MUTED)
        ax.set_title(column, fontsize=10, color=INK, loc="left", fontweight="bold")
        ax.tick_params(axis="y", colors=MUTED, labelsize=9)
    axes[0].set_ylabel(ylabel, fontsize=9, color=MUTED)
    return finish(fig, height, title, note, w_pad=2.2)


def overlaid(
    panels: Mapping[str, Mapping[str, np.ndarray]],
    marks: Union[Mapping[str, Mapping[str, float]], None] = None,
    bins: Union[np.ndarray, None] = None,
    xlabel: Union[str, None] = None,
    per_row: int = 3,
    title: Union[str, None] = None,
    note: Union[str, None] = None,
) -> Figure:
    """
    One panel per name, and in it a density histogram per series, overlaid, with vertical marks.

    - `panels`: panel -> series -> values; a series keeps its colour across panels.
    - `marks`: panel -> label -> where a vertical line goes; the label is written beside it.
    """
    bins = np.linspace(0, 1, 41) if bins is None else bins
    series = list(dict.fromkeys(name for found in panels.values() for name in found))
    palette = dict(zip(series, [NEUTRAL, *SERIES]))
    rows = -(-len(panels) // per_row)
    height = 3.4 * rows + 0.6
    fig = Figure(figsize=(4.4 * min(per_row, len(panels)) + 0.6, height), facecolor=SURFACE)
    axes = fig.subplots(rows, min(per_row, len(panels)), squeeze=False).ravel()
    for ax, (panel, found) in zip(axes, panels.items()):
        framed(ax)
        for name, values in found.items():
            values = np.asarray(values, dtype=float)
            ax.hist(values[~np.isnan(values)], bins=bins, density=True, color=palette[name], alpha=0.55, label=name)
        for depth, (label, at) in enumerate((marks or {}).get(panel, {}).items()):
            ax.axvline(at, color=INK, linewidth=1, linestyle=("-", "--", ":", "-.")[depth % 4])
            ax.annotate(f"{label} {at:.2f}", (at, 1), xycoords=("data", "axes fraction"), xytext=(4, -12 - 11 * depth), textcoords="offset points", fontsize=8, color=MUTED)
        ax.set_title(panel, fontsize=10, color=INK, loc="left", fontweight="bold")
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
        ax.legend(fontsize=8, frameon=False)
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    return finish(fig, height, title, note, w_pad=2.2, h_pad=1.6)


def strips(
    panels: Mapping[str, Mapping[str, np.ndarray]],
    marks: Union[Mapping[str, float], None] = None,
    xlabel: Union[str, None] = None,
    per_row: int = 3,
    seed: int = 0,
    title: Union[str, None] = None,
    note: Union[str, None] = None,
) -> Figure:
    """
    One panel per name, and in it a horizontal box per group with its values as points over it.

    - `panels`: panel -> group -> values, groups drawn top to bottom in the given order.
    - `marks`: panel -> where a dashed reference line goes.
    - `seed`: of the points' vertical jitter.
    """
    random = np.random.default_rng(seed)
    rows = -(-len(panels) // per_row)
    longest = max(len(found) for found in panels.values())
    height = (0.42 * longest + 1.4) * rows + 0.6
    fig = Figure(figsize=(4.6 * min(per_row, len(panels)) + 1.2, height), facecolor=SURFACE)
    axes = fig.subplots(rows, min(per_row, len(panels)), squeeze=False).ravel()
    for ax, (panel, found) in zip(axes, panels.items()):
        framed(ax)
        groups = list(found)
        data = [np.asarray(found[group], dtype=float) for group in groups]
        ax.boxplot(data, orientation="horizontal", widths=0.6, showfliers=False, patch_artist=True,
                   boxprops={"facecolor": tint(SERIES[0], 0.25), "edgecolor": EDGE}, medianprops={"color": INK}, whiskerprops={"color": EDGE}, capprops={"color": EDGE})
        for position, values in enumerate(data, start=1):
            ax.scatter(values, position + random.normal(0, 0.07, len(values)), s=4, alpha=0.35, color=SERIES[0], linewidths=0)
        if marks and panel in marks:
            ax.axvline(marks[panel], color=INK, linewidth=1, linestyle="--")
        ax.set_yticks(range(1, len(groups) + 1), wrap(groups, 22))
        ax.invert_yaxis()
        ax.set_title(panel, fontsize=10, color=INK, loc="left", fontweight="bold")
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    return finish(fig, height, title, note, w_pad=2.2, h_pad=1.6)
