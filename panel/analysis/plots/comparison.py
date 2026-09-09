"""Two distributions side by side, and one number per source."""
from typing import Mapping, Sequence, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from .style import wrap


def distribution_bars(
    predicted: pd.Series,
    reference: pd.Series,
    options: Sequence[str],
    turnout: Union[Sequence[str], None] = None,
    predicted_label: str = "Predicted",
    reference_label: str = "Reference",
    labels: Union[Mapping[str, str], None] = None,
    title: Union[str, None] = None,
) -> plt.Figure:
    """
    A predicted distribution against a reference: an attendance panel above, one horizontal bar per option below.

    Args:
        predicted, reference: series indexed by the turnout and option columns.
        options: option columns, in display order.
        turnout: the (yes, no) attendance columns; None hides the attendance panel.
        predicted_label, reference_label: legend entries.
        labels: option column -> display name; the column itself by default.
        title: figure title.
    """
    turnout = list(turnout or ())
    labels = labels or {}
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 12), gridspec_kw={"height_ratios": [1, 5]})
    if title:
        fig.suptitle(title, fontsize=20, y=0.96)

    y = np.arange(len(turnout))
    height = 0.35
    bars = top.barh(y + height / 2, [float(predicted.get(column, 0)) for column in turnout], height, label=predicted_label, color="lightgreen")
    reference_bars = top.barh(y - height / 2, [float(reference.get(column, 0)) for column in turnout], height, label=reference_label, color="darkgreen")
    top.set_yticks(y, turnout)
    top.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    top.set_xlim(0, 1)
    if turnout:
        top.legend(fontsize=12)
        top.bar_label(bars, padding=3, fmt="{:.1%}", fontsize=10)
        top.bar_label(reference_bars, padding=3, fmt="{:.1%}", fontsize=10)
    else:
        top.axis("off")

    y = np.arange(len(options))
    height = 0.4
    bars = bottom.barh(y - height / 2, [float(predicted.get(column, 0)) for column in options], height, label=predicted_label, color="skyblue")
    reference_bars = bottom.barh(y + height / 2, [float(reference.get(column, 0)) for column in options], height, label=reference_label, color="steelblue")
    bottom.set_yticks(y, wrap(labels.get(column, column) for column in options))
    bottom.invert_yaxis()
    bottom.legend(loc="lower right", fontsize=12)
    bottom.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    bottom.set_xlabel("Probability / share", fontsize=12)
    bottom.bar_label(bars, padding=3, fmt="{:.1%}", fontsize=10)
    bottom.bar_label(reference_bars, padding=3, fmt="{:.1%}", fontsize=10)
    bottom.grid(axis="x", linestyle="--", alpha=0.7)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def bars_by_source(values: pd.Series, ylabel: str, scale: float = 1.0, fmt: str = "%.1f", title: Union[str, None] = None) -> plt.Figure:
    """One bar per source (the index), values multiplied by `scale`, labelled on top."""
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(values)), 5))
    bars = ax.bar(values.index.astype(str), values.to_numpy(dtype=float) * scale, color="#4C72B0")
    ax.bar_label(bars, fmt=fmt, padding=3)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(values.index.astype(str), rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig
