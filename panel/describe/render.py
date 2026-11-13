"""
Render a dataset's descriptions over its values and store them as prompt tables.

A prompt table is one rendered description per subject, at
`data/<dataset>/prompts/<description>.csv`. It is written in the same pass as the values
table, so it cannot lag the rows it describes, and a run reads it instead of calling a
describer - the prompt a run sent is a stored artifact.
"""
from pathlib import Path
from typing import Dict, Sequence, Union

import pandas as pd

from panel.dataset import Dataset

from .forms import Forms, load_forms
from .template import PromptTemplate, load_template


def render(dataset: Dataset, template: PromptTemplate, values: pd.DataFrame, forms: Forms) -> pd.Series:
    """
    One description per row of `values`, named for the template.

    Raises:
        ValueError: if the template names a column the dataset does not have, or a row
            renders empty - an empty description is a data problem, not a short profile.
        KeyError: if a value has no form; see `Forms.phrase`.
    """
    stray = [column for column in template.columns if column not in values.columns]
    if stray:
        raise ValueError(f"description '{template.name}' names columns '{dataset.name}' does not have: {stray}")
    descriptions = values.apply(lambda row: template.render(row, forms), axis=1).rename(template.name)
    empty = descriptions.index[descriptions == ""]
    if len(empty):
        raise ValueError(
            f"description '{template.name}' renders empty for {len(empty)} rows of '{dataset.name}', e.g. ids {list(empty[:5])}"
        )
    return descriptions


def render_all(
    dataset: Dataset, values: pd.DataFrame, names: Union[Sequence[str], None] = None
) -> Dict[str, Path]:
    """
    Descriptions from the dataset's folder, rendered over `values` and stored; name -> table path.

    Args:
        names: which descriptions; None renders every one the dataset has that is rendered
            at all (a `stored` description keeps the table it has).

    Raises:
        ValueError: if a name is not one of the dataset's descriptions, or names a stored one.
    """
    templates = {name: load_template(dataset.description_path(name)) for name in dataset.descriptions}
    names = tuple(name for name in dataset.descriptions if not templates[name].stored) if names is None else tuple(names)
    unknown = [name for name in names if name not in dataset.descriptions]
    if unknown:
        raise ValueError(f"'{dataset.name}' has no descriptions {unknown}; it has {list(dataset.descriptions)}")
    stored = [name for name in names if templates[name].stored]
    if stored:
        raise ValueError(f"descriptions {stored} of '{dataset.name}' are stored as sent and are not rendered")
    forms = load_forms(dataset.forms_path)
    written = {}
    for name in names:
        descriptions = render(dataset, templates[name], values, forms)
        path = dataset.prompts_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptions.to_frame().to_csv(path, index=True, encoding="utf-8")
        written[name] = path
    return written


def read_prompts(
    dataset: Dataset, description: str, expected_index: Union[pd.Index, None] = None
) -> pd.DataFrame:
    """
    A stored prompt table, as a one-column frame to join onto the values.

    Args:
        dataset: whose rendering.
        description: which description; also the column name.
        expected_index: the subjects about to be run; coverage is checked here rather than
            discovered as a missing prompt in the middle of a paid run.

    Raises:
        FileNotFoundError: if that description has not been rendered.
        ValueError: if the table repeats a key or does not cover exactly `expected_index`.
    """
    path = dataset.prompts_path(description)
    if not path.exists():
        raise FileNotFoundError(f"no prompt table at {path}; run the dataset's prepare step first")
    table = pd.read_csv(path, index_col=0, encoding="utf-8")
    if not table.index.is_unique:
        raise ValueError(f"{path}: index has duplicates")
    if expected_index is not None:
        missing, extra = expected_index.difference(table.index), table.index.difference(expected_index)
        if len(missing) or len(extra):
            raise ValueError(
                f"{path} does not match the values: {len(missing)} subjects without a description, "
                f"{len(extra)} descriptions without a subject. Re-run the dataset's prepare step."
            )
        table = table.loc[expected_index]
    return table
