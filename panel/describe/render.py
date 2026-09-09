"""
Render a description over a dataset's profile in the language they share, and store the table.

The description's language picks the pack whose filters and `usable` predicate the template
gets; a dataset in another language is refused rather than rendered into a mixed-language
prompt. A row that fills no sentence at all is refused too: an empty description is a data
problem, not a short profile.
"""
from pathlib import Path
from typing import Callable, Dict, Mapping, Union

import pandas as pd

from panel import paths
from panel.dataset import Dataset
from panel.describe.lang import load_language
from panel.describe.tables import write_table
from panel.describe.template import PromptTemplate, load_description, render_frame


def render(
    dataset: Dataset,
    template: PromptTemplate,
    profile: pd.DataFrame,
    extras: Union[Mapping[str, Callable], None] = None,
) -> pd.Series:
    """
    One description per profile row, named for the template.

    Args:
        dataset: whose profile; its language must be the template's.
        template: the description to render.
        profile: the dataset's profile table, or a subset of it.
        extras: filters the dataset adds for its own quirks, merged over the language pack's.

    Raises:
        ValueError: if the template's language is not the dataset's, or a row renders empty.
    """
    if template.language != dataset.language:
        raise ValueError(
            f"description '{template.name}' is in '{template.language}', dataset '{dataset.name}' in '{dataset.language}'"
        )
    language = load_language(template.language)
    descriptions = render_frame(template, profile, {**language.filters, **(extras or {})}, language.usable)
    empty = descriptions.index[descriptions == ""]
    if len(empty):
        raise ValueError(
            f"description '{template.name}' renders empty for {len(empty)} rows of '{dataset.name}', e.g. ids {list(empty[:5])}"
        )
    return descriptions


def render_all(
    dataset: Dataset, profile: pd.DataFrame, descriptions_dir: Union[str, Path] = paths.DESCRIPTIONS_DIR
) -> Dict[str, Path]:
    """Every description the dataset declares, rendered over `profile` and stored; name -> table path."""
    return {
        name: write_table(dataset.name, render(dataset, load_description(name, descriptions_dir), profile))
        for name in dataset.descriptions
    }
