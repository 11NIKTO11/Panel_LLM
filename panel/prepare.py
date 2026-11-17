"""
Prepare a dataset: build its values table, render its descriptions.

The general half of the prepare step. What is specific to a dataset is one function in its
folder, `datasets/<name>/build.py::build(dataset) -> DataFrame`, which turns the raw source
into values; everything after that - checking, storing, rendering - is the same for all.

A saved run records the sha256 of the tables it was built from, and reading it back refuses
values that have changed since. So prepare does not replace such a table with a different one
unless told to: it writes, compares, and on a refusal puts every table back as it was.
"""
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Union

from panel import paths
from panel.dataset import Dataset, load_dataset, read_values, write_values
from panel.describe.render import read_prompts, render_all
from panel.digest import text, text_sha256
from panel.run.store import digests_in_use

VALUES = "values"


@dataclass(frozen=True)
class PromptCount:
    """`identical` counts the prompts that are word for word another subject's, `distinct` the different texts among all."""
    prompts: int
    identical: int
    distinct: int


def _stored(dataset: Dataset) -> Dict[Path, bytes]:
    """Every table prepare has written for the dataset, as it stands."""
    tables = [dataset.values_path, *dataset.prompts_path("*").parent.glob("*.csv")]
    return {path: path.read_bytes() for path in tables if path.exists()}


def _refuse_replaced(before: Dict[Path, bytes], written: Sequence[Path], in_use: Dict[str, List[str]]) -> None:
    """Put the tables back and raise, if writing changed one a saved run was built from."""
    runs = {
        path: in_use.get(text_sha256(before[path]), [])
        for path in written if path in before and text(before[path]) != text(path.read_bytes())
    }
    refused = {path: stems for path, stems in runs.items() if stems}
    if not refused:
        return
    for path in written:
        if path in before:
            path.write_bytes(before[path])
        else:
            path.unlink()
    listed = "; ".join(f"{path} ({len(stems)} runs, e.g. {stems[0]})" for path, stems in refused.items())
    raise FileExistsError(
        f"prepare would replace tables that saved runs were built from: {listed}. Nothing was changed. "
        "New wording belongs in a description of a new name; with overwrite those runs can no longer "
        "be read against their dataset."
    )


def prepare(
    name: str, descriptions: Union[Sequence[str], None] = None, rebuild_values: bool = True,
    overwrite: bool = False, results_dir: Union[str, Path] = paths.RESULTS_DIR,
) -> Dict[str, Path]:
    """
    What was written: `values` (when rebuilt) and each description name -> its path.

    `descriptions=None` renders all of them; `rebuild_values=False` renders over the stored values,
    for when only a template or a form changed; `results_dir` is where the saved runs are looked for.
    """
    dataset = load_dataset(name)
    before = _stored(dataset)
    written = {}
    if rebuild_values:
        if dataset.source is None:
            raise ValueError(f"'{name}' is prepared data only: it has no source to rebuild its values from")
        builder = importlib.import_module(f"datasets.{name}.build")
        written[VALUES] = write_values(dataset, builder.build(dataset))
    written.update(render_all(dataset, read_values(dataset), descriptions))
    if not overwrite:
        _refuse_replaced(before, list(written.values()), digests_in_use(results_dir))
    return written


def count_prompts(name: str, description: str) -> PromptCount:
    prompts = read_prompts(load_dataset(name), description)[description]
    return PromptCount(len(prompts), int(prompts.duplicated(keep=False).sum()), int(prompts.nunique()))
