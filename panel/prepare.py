"""
Prepare a dataset: build its values table, render its descriptions.

The general half of the prepare step. What is specific to a dataset is one function in its
folder, `datasets/<name>/build.py::build(dataset) -> DataFrame`, which turns the raw source
into values; everything after that - checking, storing, rendering - is the same for all.
"""
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Union

from panel.dataset import load_dataset, read_values, write_values
from panel.describe.render import read_prompts, render_all

VALUES = "values"


@dataclass(frozen=True)
class PromptCount:
    """
    How far a prompt table tells its subjects apart.

    Attributes:
        prompts: one per subject.
        identical: prompts that are word for word another subject's.
        distinct: different texts among them.
    """
    prompts: int
    identical: int
    distinct: int


def prepare(
    name: str, descriptions: Union[Sequence[str], None] = None, rebuild_values: bool = True
) -> Dict[str, Path]:
    """
    Write a dataset's prepared output under `data/<name>/`.

    Args:
        name: the dataset, a folder under `datasets/`.
        descriptions: which of its descriptions to render; None renders all of them.
        rebuild_values: build the values table from the raw source; False renders over the
            stored one, for when only a template or a form changed.

    Returns:
        What was written: `values` (when rebuilt) and each description name -> its path.

    Raises:
        FileNotFoundError: if the dataset, its builder or - with `rebuild_values=False` - its
            stored values table is missing.
        ValueError: if the values are not what the dataset declares, or a description is unknown.
    """
    dataset = load_dataset(name)
    written = {}
    if rebuild_values:
        builder = importlib.import_module(f"datasets.{name}.build")
        written[VALUES] = write_values(dataset, builder.build(dataset))
    written.update(render_all(dataset, read_values(dataset), descriptions))
    return written


def count_prompts(name: str, description: str) -> PromptCount:
    """
    Count the identical prompts of a stored prompt table.

    Raises:
        FileNotFoundError: if the dataset or that prompt table is missing.
    """
    prompts = read_prompts(load_dataset(name), description)[description]
    return PromptCount(len(prompts), int(prompts.duplicated(keep=False).sum()), int(prompts.nunique()))
