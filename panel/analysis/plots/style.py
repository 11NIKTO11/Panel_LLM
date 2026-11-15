"""Shared looks: the surface, a panel fill and edge, inks, a fixed colour per named series, two neutrals for
what the series are held against (the reference and a second yardstick), a colour range
per kind of heatmap (errors run hot, effect sizes in the legacy figure's range, agreement green,
shifts orange, distances purple), label wrapping,
the framed panel, and the title and note every figure ends with."""
import textwrap
from typing import Dict, Iterable, List, Tuple, Union

from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
PANEL = "#f5f4f1"
EDGE = "#b9b8b2"
NEUTRAL = "#6f6e69"
REFERENCE = "#1b1b1b"
SECONDARY = "#a3a19b"
ERRORS = "YlOrRd"
EFFECTS = "cividis"
AGREEMENT = "Greens"
SHIFTS = "Oranges"
DISTANCES = "Purples"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")


def colors(names: Iterable[str]) -> Dict[str, str]:
    """
    A colour per name, in the palette's order. Pass every name a series can have, not only
    those drawn, so a series keeps its colour when another is absent.

    Raises:
        ValueError: on more names than the palette has colours.
    """
    names = list(names)
    if len(names) > len(SERIES):
        raise ValueError(f"{len(names)} series, the palette has {len(SERIES)} colours")
    return dict(zip(names, SERIES))


def legible(color: Tuple[float, ...]) -> str:
    """The ink that reads on a cell of this colour: the surface on a dark one, ink on a light one."""
    red, green, blue = color[:3]
    return SURFACE if 0.299 * red + 0.587 * green + 0.114 * blue < 0.55 else INK


def tint(color: str, strength: float) -> Tuple[float, float, float]:
    """The colour mixed into the surface: 1 is the colour itself, 0 the surface."""
    return tuple(strength * c + (1 - strength) * s for c, s in zip(to_rgb(color), to_rgb(SURFACE)))


def wrap(labels: Iterable[str], width: int = 26) -> List[str]:
    """Each label broken into lines of at most `width` characters."""
    return [textwrap.fill(str(label), width=width) for label in labels]


def clean(ax) -> None:
    """Axes on the shared surface: no frame, a faint grid along the value axis, muted ticks."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, colors=MUTED, labelsize=9)
    ax.tick_params(axis="y", colors=INK, labelsize=10)


def framed(ax) -> None:
    """Axes as a panel of their own: `clean`, on the panel fill, with an edge all round."""
    clean(ax)
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(EDGE)


def finish(
    fig: Figure, height: float, title: Union[str, None] = None, note: Union[str, None] = None,
    bottom: float = 0.0, noted: float = 0.5, tight: bool = True, **pads: float,
) -> Figure:
    """
    The title, top left and anchored by its top, the note under it, and the axes laid out below them.

    Args:
        fig: the figure, its axes drawn.
        height: of the figure, in inches; everything here is placed in inches from an edge.
        title: figure title.
        note: a line under the title.
        bottom: inches kept free under the axes, for a legend.
        noted: inches kept free above the axes when there is a note; 0.3 without one.
        tight: lay the axes out; off for a figure whose colour bar spans its axes.
        pads: `w_pad`, `h_pad` of the layout.
    """
    if title:
        fig.suptitle(title, x=0.02, y=1 - 0.1 / height, ha="left", va="top", fontsize=13, color=INK, fontweight="bold")
    if note:
        fig.text(0.02, 1 - 0.42 / height, note, ha="left", va="top", fontsize=9, color=MUTED)
    if tight:
        fig.tight_layout(rect=(0, bottom / height, 1, 1 - (noted if note else 0.3) / height), **pads)
    return fig
