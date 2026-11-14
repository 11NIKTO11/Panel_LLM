"""
A grid's values: every combination of the levels its declaration lists.

A grid dataset has no raw source; its `dataset.json` is the source. Each column is a factor
and its `levels.order` the factor's levels, so the same lists that order the levels in
analysis are the ones the grid is built from. A grid's `build.py` is this module's
`full_factorial`, which is how `panel.prepare` reaches it.
"""
import itertools

import pandas as pd

from panel.dataset import INDEX, Dataset


def full_factorial(dataset: Dataset) -> pd.DataFrame:
    """
    One values row per combination of the columns' levels, numbered from 1, the first column
    varying slowest.

    Raises:
        ValueError: if a column declares no order of levels.
    """
    unordered = [column for column in dataset.columns if column not in dataset.levels or not dataset.levels[column].order]
    if unordered:
        raise ValueError(f"grid '{dataset.name}': columns without an order of levels: {unordered}")
    combinations = itertools.product(*(dataset.levels[column].order for column in dataset.columns))
    values = pd.DataFrame(list(combinations), columns=list(dataset.columns))
    values.index = pd.RangeIndex(1, len(values) + 1, name=INDEX)
    return values
