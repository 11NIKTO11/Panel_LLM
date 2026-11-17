"""
A saved run read back, with everything an analysis sets beside it.

The record names the run's topic and dataset, so a stem is enough. Every analysis takes a
`SavedRun` and reads the same four things from it:

- `results` / `normalized` - one row per call, indexed by subject id, the topic's columns;
- `subjects` - the dataset's values for exactly the subjects the run asked, one row each;
- `claims` - how the dataset's own answer to the topic sorts into the topic's options;
- `reference` - the topic's real-world results, one row per area.

`subjects` is the run's subjects and not the whole values table on purpose: a run over a
sample is compared with what that sample claimed. `values` holds the whole table.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd

from panel import paths
from panel.dataset import Claims, Dataset, load_claims, load_dataset, read_values
from panel.digest import content_sha256, recorded
from panel.run import ResultStore, RunSpec
from panel.run.resolve import REPEAT
from panel.run.store import PROMPT
from panel.topic import PlotStyle, Positions, Topic, load_plot_style, load_positions, load_topic


class NotApplicable(Exception):
    """An analysis has nothing to work on for this run, e.g. no reference to measure against."""


@dataclass(frozen=True)
class SavedRun:
    """
    `results` and `normalized` hold `repeat` before the topic's columns if the run repeats;
    `subjects` is in the order of `results`. `style` is empty when the topic has none or none was
    asked for. `prompts` is the prompt of every row of `results`, as sent. `weighted`: a predicted distribution weights each answer's option shares by its
    probability of voting; off, every answer with a preference counts the same.
    """
    stem: str
    spec: RunSpec
    record: Dict[str, Any]
    topic: Topic
    dataset: Dataset
    results: pd.DataFrame
    normalized: pd.DataFrame
    subjects: pd.DataFrame
    values: pd.DataFrame
    claims: Union[Claims, None]
    reference: Union[pd.DataFrame, None]
    style: PlotStyle
    positions: Union[Positions, None] = None
    weighted: bool = True
    prompts: Union[pd.Series, None] = None

    @property
    def options(self) -> List[str]:
        return list(self.topic.options)

    @property
    def labels(self) -> Dict[str, str]:
        """Topic column -> the name a figure shows: the style's label, else the column itself, an option's id being its short name."""
        return {**{column: column for column in self.topic.columns}, **self.style.labels}

    @property
    def title(self) -> str:
        return f"{self.spec.model} on {self.topic.name}"

    @property
    def total(self) -> Union[str, None]:
        """The reference row that is the whole, or None without a reference."""
        return self.topic.reference.total if self.topic.reference else None


def pools(run: SavedRun, analysis: str) -> None:
    """Called by an analysis that reads the subjects as a sample of one population."""
    if run.dataset.aggregate:
        raise NotApplicable(f"'{run.stem}': a subject of '{run.dataset.name}' is already a whole, and {analysis} pools subjects")


def find_stem(store: ResultStore, text: str) -> str:
    """The one saved run whose stem is `text` or contains it."""
    stems = store.runs()
    if text in stems:
        return text
    found = [stem for stem in stems if text in stem]
    if len(found) != 1:
        raise ValueError(f"'{text}' names {len(found)} runs of '{store.topic}': {found or stems}")
    return found[0]


def load_run(
    store: ResultStore, stem: str, styled: bool = True, weighted: bool = True,
    datasets_dir: Union[str, Path] = paths.DATASETS_DIR, topics_dir: Union[str, Path] = paths.TOPICS_DIR,
) -> SavedRun:
    """`styled` draws with the topic's `plot.json`; off, figures use their own palette and the native names."""
    record = store.record(stem)
    if record.get("incomplete"):
        raise ValueError(f"'{stem}' is an incomplete run, {len(record['missing'])} of its calls missing; it is not analysed")
    spec = RunSpec.from_dict(record["spec"])
    topic, dataset = load_topic(spec.topic, topics_dir), load_dataset(spec.dataset, datasets_dir)

    if content_sha256(dataset.values_path) != recorded(record["dataset"])["values"]:
        raise ValueError(f"{dataset.values_path} was rebuilt after '{stem}' was sent; its subjects can no longer be trusted to match")
    values = read_values(dataset)

    columns = ([REPEAT] if spec.repeats > 1 else []) + topic.columns
    saved = topic.rename(store.frame(stem))
    results, normalized = saved[columns], topic.rename(store.frame(stem, normalized=True))[columns]
    rows = record["dataset"]["rows"]
    if set(results.index) != set(rows) or not results.index.equals(normalized.index):
        raise ValueError(f"'{stem}': the saved rows are not the {len(rows)} subjects its record lists")
    subjects = values.loc[results.index.drop_duplicates()]

    reference = topic.load_reference().set_index(topic.reference.key) if topic.reference else None
    if reference is not None and dataset.area:
        unknown = sorted(set(subjects[dataset.area]) - set(reference.index))
        if unknown:
            raise ValueError(f"'{stem}': areas of '{dataset.area}' with no reference row: {unknown}")

    return SavedRun(
        stem=stem, spec=spec, record=record, topic=topic, dataset=dataset, results=results,
        normalized=normalized, subjects=subjects, values=values,
        claims=load_claims(dataset, topic.name) if topic.name in dataset.claims else None,
        reference=reference, style=load_plot_style(topic) if styled else PlotStyle(), positions=load_positions(topic), weighted=weighted,
        prompts=saved[PROMPT] if PROMPT in saved and saved[PROMPT].notna().all() else None,
    )


def describe(run: SavedRun) -> str:
    """What was loaded, in a few lines: the run, its size, and what there is to compare it with."""
    spec = run.spec
    lines = [
        run.stem,
        f"  topic      {run.topic.name}: {len(run.options)} options, {'with' if run.topic.turnout else 'no'} turnout",
        f"  model      {spec.client} {spec.model}, {spec.shape}, saved {run.record['saved_at']}",
        f"  subjects   {len(run.subjects)} of {len(run.values)} in '{run.dataset.name}', {len(run.results)} answers"
        + (f", {len(run.record['faulty'])} kept though flawed" if run.record.get("faulty") else ""),
        f"  claimed    {'column ' + repr(run.claims.column) if run.claims else 'none'}",
        f"  actual     {f'{len(run.reference) - 1} areas and {run.total!r}' if run.reference is not None else 'none'}",
        f"  area       {run.dataset.area or 'none'}",
        f"  groups     {', '.join(run.dataset.groups) or 'none'}",
        f"  predicted  {'option shares weighted by the probability of voting' if run.weighted else 'option shares not weighted by turnout'}",
    ]
    return "\n".join(lines)
