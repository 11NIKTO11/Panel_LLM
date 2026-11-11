"""
A grouping column read as levels, the way its dataset declares (`panel.dataset.Levels`):

- `order`: the column's values along their scale, as declared;
- `bands`: a numeric column cut into the declared bands;
- `several`: a cell may hold several answers, and the subject belongs to each of them, so
  the levels' subjects add up to more than the subjects;
- no declaration: the column's values, the largest group first.
"""
from typing import Dict, Union

import pandas as pd

from panel.dataset import Levels


def members(values: pd.Series, levels: Union[Levels, None] = None) -> Dict[str, pd.Index]:
    """
    Level -> the subjects at it, in display order; a level nobody is at is left out.

    Args:
        values: one grouping column, indexed by subject; missing values belong nowhere.
        levels: the dataset's declaration for the column, or None.

    Raises:
        ValueError: on a value the declared order does not hold or above the last band.
    """
    values = values.dropna()
    if levels and levels.several:
        held = values.str.split(levels.several, regex=False).explode()
        return {name: held.index[held == name] for name in held.value_counts().index}
    if levels and levels.bands:
        labels, tops = list(levels.bands), list(levels.bands.values())
        if (values > tops[-1]).any():
            raise ValueError(f"'{values.name}' has values above its last band, {tops[-1]}")
        cut = pd.cut(values, [float("-inf"), *tops], labels=labels)
        return {name: cut.index[cut == name] for name in labels if (cut == name).any()}
    if levels and levels.order:
        unknown = sorted(set(values) - set(levels.order))
        if unknown:
            raise ValueError(f"'{values.name}' has values its declared order does not hold: {unknown}")
        return {name: values.index[values == name] for name in levels.order if (values == name).any()}
    values = values.astype(str)
    return {name: values.index[values == name] for name in values.value_counts().index}
