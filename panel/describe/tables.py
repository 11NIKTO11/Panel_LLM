"""
Prompt tables: a rendered description, stored per dataset at `data/prompts/<dataset>/<description>.csv`.

A description is data, not a runtime callable. A dataset's prepare step renders the
descriptions it declares, once, and a run then names the table and reads the column, so the
prompt a run sent is a stored artifact rather than the output of whatever the describer was
that day. A variant is additive: different wording means a new description name, never an
overwrite, so a recorded name keeps meaning what it meant.
"""
from pathlib import Path
from typing import Union

import pandas as pd

from panel import paths


def table_path(dataset: str, description: str, prompts_dir: Union[str, Path] = paths.PROMPTS_DIR) -> Path:
    """Where one dataset's rendering of one description lives."""
    return Path(prompts_dir) / dataset / f"{description}.csv"


def write_table(dataset: str, descriptions: pd.Series, prompts_dir: Union[str, Path] = paths.PROMPTS_DIR) -> Path:
    """Store one rendered description for one dataset, keyed by the profile index; the series' name is the description's."""
    path = table_path(dataset, descriptions.name, prompts_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptions.to_frame().to_csv(path, index=True, encoding="utf-8")
    return path


def read_table(
    dataset: str,
    description: str,
    expected_index: Union[pd.Index, None] = None,
    prompts_dir: Union[str, Path] = paths.PROMPTS_DIR,
) -> pd.DataFrame:
    """
    A rendered description, as a one-column frame to join onto the profile.

    Args:
        dataset: whose rendering.
        description: which description; also the column name.
        expected_index: the index of the rows about to be run. Given it, coverage is checked
            here rather than discovered as a NaN prompt in the middle of a paid run.

    Raises:
        FileNotFoundError: if that description has not been rendered for that dataset.
        ValueError: if the table does not cover exactly `expected_index`, or repeats a key.
    """
    path = table_path(dataset, description, prompts_dir)
    if not path.exists():
        raise FileNotFoundError(f"no prompt table at {path}; run the dataset's prepare step first")

    table = pd.read_csv(path, index_col=0, encoding="utf-8")
    if not table.index.is_unique:
        raise ValueError(f"{path}: index has duplicates")
    if expected_index is not None:
        missing = expected_index.difference(table.index)
        extra = table.index.difference(expected_index)
        if len(missing) or len(extra):
            raise ValueError(
                f"{path} does not match the profile: {len(missing)} rows without a description, "
                f"{len(extra)} descriptions without a row. Re-run the dataset's prepare step."
            )
        table = table.loc[expected_index]
    return table
