"""Shared looks: a stable colour per named series, and label wrapping for long option names."""
import textwrap
from typing import Dict, Iterable, List

PALETTE = ("#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD", "#8C8C8C", "#937860")


def colors(names: Iterable[str]) -> Dict[str, str]:
    """A colour per name, in order, cycling through the palette."""
    return {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(names)}


def wrap(labels: Iterable[str], width: int = 21) -> List[str]:
    """Each label broken into lines of at most `width` characters."""
    return [textwrap.fill(str(label), width=width) for label in labels]
