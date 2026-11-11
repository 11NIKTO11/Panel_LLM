"""The areas' values: the `region` column of the source named in `dataset.json`, one row per area."""
import pandas as pd

from panel.dataset import INDEX, Dataset


def build(dataset: Dataset) -> pd.DataFrame:
    """One values row per area, numbered from 1 in the source's order."""
    values = pd.read_csv(dataset.dir / dataset.source, encoding="utf-8")[list(dataset.columns)]
    values.index = pd.RangeIndex(1, len(values) + 1, name=INDEX)
    return values
