"""Rescaling a results frame so each of a topic's column groups sums to 1 per row."""
from typing import Sequence

import pandas as pd


def normalize_groups(df: pd.DataFrame, groups: Sequence[Sequence[str]]) -> pd.DataFrame:
    out = df.copy()
    for columns in groups:
        columns = list(columns)
        out[columns] = out[columns].div(out[columns].sum(axis=1), axis=0)
    return out
