"""Lines through points: one line per named series on shared axes."""
from typing import Mapping, Sequence, Union

import pandas as pd
from matplotlib.figure import Figure

from .style import INK, MUTED, SERIES, SURFACE, clean


def curves(
    lines: Mapping[str, pd.DataFrame],
    x: str,
    y: str,
    colors: Union[Mapping[str, str], None] = None,
    diagonal: bool = False,
    xlabel: Union[str, None] = None,
    ylabel: Union[str, None] = None,
    title: Union[str, None] = None,
    note: Union[str, None] = None,
) -> Figure:
    """
    One line with a dot per row for every series.

    Args:
        lines: series name -> its points, in the order they are joined.
        x: the column along the horizontal axis.
        y: the column along the vertical axis.
        colors: series name -> colour; the palette in order by default.
        diagonal: draw the dotted line where both are equal, and give both axes one scale.
        xlabel: horizontal axis label.
        ylabel: vertical axis label.
        title: figure title.
        note: a line under the title.
    """
    colors = colors or {}
    height = 5.6
    fig = Figure(figsize=(6.4, height), facecolor=SURFACE)
    ax = fig.subplots()
    clean(ax)
    ax.grid(axis="y", color=ax.xaxis.get_gridlines()[0].get_color(), linewidth=0.8)
    for number, (name, points) in enumerate(lines.items()):
        ax.plot(points[x], points[y], marker="o", markersize=4, linewidth=1.6, color=colors.get(name, SERIES[number % len(SERIES)]), label=name)
    if diagonal:
        top = max(float(points[[x, y]].to_numpy().max()) for points in lines.values())
        ax.plot([0, top], [0, top], linestyle=":", linewidth=1, color=MUTED)
        ax.set_aspect("equal")
    ax.set_xlabel(xlabel or x, fontsize=10, color=INK)
    ax.set_ylabel(ylabel or y, fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    if title:
        fig.suptitle(title, x=0.02, ha="left", fontsize=13, color=INK, fontweight="bold")
    if note:
        fig.text(0.02, 1 - 0.42 / height, note, ha="left", va="top", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 1 - (0.5 if note else 0.3) / height))
    return fig


def line_panels(
    tables: Mapping[str, pd.DataFrame],
    levels: Union[Mapping[str, pd.Series], None] = None,
    colors: Union[Mapping[str, str], None] = None,
    percent: Sequence[str] = (),
    zero: Sequence[str] = (),
    per_row: int = 2,
    unit: str = "pp",
    title: Union[str, None] = None,
    note: Union[str, None] = None,
) -> Figure:
    """
    One panel per measure; in each, a line per series across the same places.

    Args:
        tables: measure -> its table, indexed by place in the order drawn, a column per series.
        levels: measure -> a value per series, drawn as that series' dashed level line
            (e.g. its value for the whole).
        colors: series -> colour; the palette in order by default.
        percent: measures holding shares, shown times 100, the panel title saying `unit`.
        zero: measures that are signed, drawn with a line at zero.
        per_row: panels in a row of the grid.
        unit: what a `percent` measure is in.
        title: figure title.
        note: a line under the title.
    """
    levels, colors = levels or {}, colors or {}
    across = min(per_row, len(tables))
    down = -(-len(tables) // across)
    first = next(iter(tables.values()))
    height = 1.3 + 2.9 * down + 0.07 * max(len(str(place)) for place in first.index)
    fig = Figure(figsize=(0.42 * len(first) * across + 3.4, height), facecolor=SURFACE)
    axes = fig.subplots(down, across, sharex=True, squeeze=False)
    for ax in axes.ravel()[len(tables):]:
        ax.set_visible(False)
    x = range(len(first))
    for ax, (measure, table) in zip(axes.ravel(), tables.items()):
        clean(ax)
        ax.grid(axis="y", color=ax.xaxis.get_gridlines()[0].get_color(), linewidth=0.8)
        ax.grid(axis="x", visible=False)
        scale = 100 if measure in percent else 1
        for number, series in enumerate(table.columns):
            color = colors.get(series, SERIES[number % len(SERIES)])
            ax.plot(x, table[series].to_numpy(dtype=float) * scale, marker="o", markersize=3.5, linewidth=1.5, color=color, label=str(series))
            if measure in levels and series in levels[measure].index:
                ax.axhline(float(levels[measure][series]) * scale, color=color, linewidth=1, linestyle=(0, (4, 3)), alpha=0.7)
        if measure in zero:
            ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_title(str(measure) + (f" ({unit})" if measure in percent else ""), fontsize=10, color=INK, loc="left", fontweight="bold")
        ax.set_xticks(list(x), [str(place) for place in first.index], rotation=40, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="x", labelsize=9, colors=INK)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=min(len(names), 4), frameon=False, fontsize=9)
    if title:
        fig.suptitle(title, x=0.02, ha="left", fontsize=13, color=INK, fontweight="bold")
    if note:
        fig.text(0.02, 1 - 0.42 / height, note, ha="left", va="top", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, (0.3 + 0.25 * -(-len(names) // 4)) / height, 1, 1 - (0.6 if note else 0.3) / height), w_pad=2.0, h_pad=1.4)
    return fig
