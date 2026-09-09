"""How per-subject distributions are spread: per option, around their mean, and as sum errors."""
from typing import Mapping, Sequence, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns


def option_distribution(
    frame: pd.DataFrame,
    options: Sequence[str],
    title: Union[str, None] = None,
    kind: str = "box",
    add_points: bool = True,
    sample: Union[int, None] = None,
    point_size: float = 2.0,
    alpha: float = 0.3,
    labels: Union[Mapping[str, str], None] = None,
    xlabel: str = "option",
) -> plt.Figure:
    """
    The distribution of each option's probability over subjects, one box or violin per option.

    Args:
        frame: one row per subject with the option columns.
        options: option columns, in display order.
        kind: "box" or "violin".
        add_points: overlay the subjects as jittered points.
        sample: draw this many rows first, to keep the overlay legible.
        labels: option column -> display name.

    Raises:
        ValueError: on an unknown kind.
    """
    if kind not in ("box", "violin"):
        raise ValueError(f"kind must be 'box' or 'violin', not {kind!r}")
    labels = labels or {}
    values = frame[list(options)].clip(lower=0, upper=1)
    if sample is not None and len(values) > sample:
        values = values.sample(sample, random_state=42)
    long = values.reset_index(drop=True).melt(var_name=xlabel, value_name="value")
    long[xlabel] = long[xlabel].map(lambda column: labels.get(column, column))

    fig, ax = plt.subplots(figsize=(max(10, int(len(options) * 0.9)), 10))
    if kind == "violin":
        sns.violinplot(data=long, x=xlabel, y="value", ax=ax, inner=None, cut=0, linewidth=1, color="#89CFF0")
    else:
        sns.boxplot(data=long, x=xlabel, y="value", ax=ax, showfliers=False, color="#89CFF0")
    if add_points:
        sns.stripplot(data=long, x=xlabel, y="value", ax=ax, color="black", size=point_size, alpha=alpha, jitter=0.2)
    ax.set_ylabel("Probability")
    ax.set_xlabel(xlabel)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=90)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def error_histogram(errors: Sequence[float], title: Union[str, None] = None, xlabel: str = "absolute error from 1") -> plt.Figure:
    """A histogram of how far each flagged row's probabilities sum from 1."""
    errors = np.asarray(list(errors), dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(errors, bins=min(50, max(10, int(np.sqrt(max(len(errors), 1))))), color="#1f77b4", edgecolor="white")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("rows")
    ax.grid(axis="y", alpha=0.2)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def spread_histograms(frame: pd.DataFrame, columns: Sequence[str], title: Union[str, None] = None) -> plt.Figure:
    """
    How far the subjects' distributions sit from their mean: histograms of Euclidean distance and cosine distance.
    """
    values = frame[list(columns)].to_numpy(dtype=float)
    mean = values.mean(axis=0)
    euclidean = np.linalg.norm(values - mean, axis=1)
    norms = np.linalg.norm(values, axis=1) * np.linalg.norm(mean)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine = 1 - np.where(norms > 0, values @ mean / norms, np.nan)
    fig, (left, right) = plt.subplots(1, 2, figsize=(16, 6))
    sns.histplot(euclidean, ax=left, kde=True)
    left.set_title("Euclidean distance from the mean")
    left.set_xlabel("distance")
    left.set_ylabel("subjects")
    sns.histplot(cosine[~np.isnan(cosine)], ax=right, kde=True)
    right.set_title("Cosine distance from the mean")
    right.set_xlabel("1 - cosine similarity")
    right.set_ylabel("subjects")
    if title:
        fig.suptitle(title, fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig
