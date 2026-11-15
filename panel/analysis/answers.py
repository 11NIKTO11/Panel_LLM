"""
What a run's answers say on average, compared with nothing: the first reading of a run.

The numbers are `panel.run.check.means`, the ones `run.py` prints after a run; here they are
kept as a table. There is no figure.
"""
from typing import Dict, List, Tuple

import pandas as pd
from matplotlib.figure import Figure

from panel.run.check import means

from .output import Output, save
from .saved import SavedRun

NAME = "answers"


def compute(run: SavedRun) -> Dict[str, pd.DataFrame]:
    """`answers` - the mean of every topic column and how often each option is an answer's highest."""
    return {NAME: means(run.results, run.topic)}


def draw(run: SavedRun, tables: Dict[str, pd.DataFrame]) -> Dict[str, Figure]:
    """No figure."""
    return {}


def answers(run: SavedRun, out: Output) -> Tuple[pd.DataFrame, List[Figure]]:
    """
    The mean of every topic column and how often each option is an answer's highest.

    Args:
        run: the saved run.
        out: takes the table, named `answers`.

    Returns:
        The table of `compute`, and no figure.
    """
    tables = compute(run)
    return save(out, tables, draw(run, tables))
