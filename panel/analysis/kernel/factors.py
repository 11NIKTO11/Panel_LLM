"""
A kernel's factors as columns of a design: an indicator per level, the way the dataset declares
the factor (bands, several answers per cell), for the regressions that explain the answers.
"""
from typing import Dict, Sequence, Union

import numpy as np
import pandas as pd

from ..levels import banded
from ..stats.design import indicators
from .matrix import MISSING, Kernel


def design(kernel: Kernel, columns: Union[Sequence[str], None] = None) -> Dict[str, np.ndarray]:
    """Factor -> its indicator columns over the profiles, one per level (bands and several answers as declared)."""
    out = {}
    for column in columns or kernel.factors.columns:
        declared = kernel.levels.get(column)
        values = kernel.factors[column].replace(MISSING, np.nan) if declared and declared.several else kernel.factors[column]
        out[column] = indicators(banded(values, declared), declared.several if declared else None, MISSING)
    return out


def level_designs(kernel: Kernel, columns: Union[Sequence[str], None] = None) -> Dict[str, pd.DataFrame]:
    """Factor -> its design as a table: a row per level the profiles hold (the raw value), its indicator columns."""
    return {
        column: pd.DataFrame(block, index=pd.Index(kernel.factors[column].to_numpy(), name=column)).groupby(level=0, sort=True).first()
        for column, block in design(kernel, columns).items()
    }


def players(kernel: Kernel) -> Dict[str, np.ndarray]:
    """
    The factors as players of an attribution: name -> design columns; the columns the
    dataset declares `together` are one player under its name, each other factor its own.
    """
    blocks = design(kernel)
    out, done = {}, set()
    for name, group in kernel.run.dataset.together.items():
        held = [column for column in group if column in blocks]
        if held:
            out[name] = np.hstack([blocks[column] for column in held])
            done |= set(held)
    return {**{column: blocks[column] for column in blocks if column not in done}, **out}
