"""
How subjects are grouped: by a profile column within one dataset, or by a common level across datasets.

`by` is the one group-by. Results join the profile on subject id, so any profile column is a
grouping key; region is the first caller. Across datasets raw columns cannot be grouped
together (a CZK band is not a `low`), so the schema declares a common scale per concept and
`to_levels` maps whichever representation a profile holds onto it.
"""
from typing import Callable, Dict, Union

import numpy as np
import pandas as pd

from panel.schema import Schema

from .aggregate import aggregate_results_from_df, claimed_distribution


def by(
    frame: pd.DataFrame,
    key: str,
    aggregate: Callable[[pd.DataFrame], pd.Series],
    total: Union[str, None] = None,
) -> pd.DataFrame:
    """
    One aggregated row per value of `frame[key]`, in sorted order, plus a `total` row over everything.

    Args:
        frame: rows to aggregate, already carrying the grouping column.
        key: the grouping column.
        aggregate: turns a group of rows into one distribution.
        total: label of the row aggregating every row; omitted when None.

    Raises:
        KeyError: if the frame lacks the grouping column.
    """
    groups = {value: aggregate(part) for value, part in frame.groupby(key, sort=True, observed=True)}
    table = pd.DataFrame(groups).T
    if total is not None:
        table.loc[total] = aggregate(frame)
    table.index.name = key
    return table


def predicted_by(
    results: pd.DataFrame,
    profile: pd.DataFrame,
    key: str,
    topic,
    total: Union[str, None] = None,
    weighted: bool = True,
    normalized: bool = True,
) -> pd.DataFrame:
    """
    Predicted distributions per group: results joined with the profile's grouping column on id.

    Raises:
        ValueError: if a results row has no profile row.
    """
    joined = results.join(profile[[key]], how="inner")
    if len(joined) != len(results):
        raise ValueError(f"{len(results) - len(joined)} results rows have no profile row")
    return by(joined, key, lambda part: aggregate_results_from_df(part, topic, weighted, normalized), total)


def claimed_by(
    profile: pd.DataFrame,
    key: str,
    claims,
    topic,
    total: Union[str, None] = None,
    count_excluded_as_abstained: bool = False,
) -> pd.DataFrame:
    """Claimed distributions per group of the profile."""
    return by(profile, key, lambda part: claimed_distribution(part, claims, topic, count_excluded_as_abstained), total)


def spread_by(frame: pd.DataFrame, key: str, columns, ddof: int = 1) -> pd.DataFrame:
    """Standard deviation of each column within each group: rows are columns, groups are columns."""
    return frame.groupby(key, sort=True, observed=True)[list(columns)].std(ddof=ddof).T


def to_levels(profile: pd.DataFrame, concept: str, schema: Schema) -> pd.Series:
    """
    The profile's rows on the concept's common scale, from whichever representation it holds.

    The first column of the concept that the profile carries and the schema knows how to map
    decides: numeric `bins` for an exact value, a `map` for a band or phrase.

    Raises:
        ValueError: if the concept has no common scale, the profile carries no mappable column
            of it, or a value falls outside the map.
    """
    declared = schema.concepts[concept]
    if not declared.levels:
        raise ValueError(f"concept '{concept}' declares no common levels")
    for column in declared.columns:
        if column.name not in profile.columns or (column.bins is None and column.map is None):
            continue
        values = profile[column.name]
        if values.isna().any():
            raise ValueError(f"column '{column.name}' has {int(values.isna().sum())} missing values")
        if column.bins is not None:
            edges = [-np.inf, *column.bins, np.inf]
            cut = pd.cut(values.astype(float), bins=edges, right=False, labels=list(declared.levels))
            return pd.Series(cut.astype(str), index=profile.index, name=concept)
        mapped = values.map(column.map)
        unknown = sorted(set(values[mapped.isna()].astype(str)))
        if unknown:
            raise ValueError(f"column '{column.name}' has values outside its map onto {concept} levels: {unknown}")
        return mapped.rename(concept)
    raise ValueError(f"profile carries no column of concept '{concept}' that maps onto its levels")
