"""
Several saved runs of one topic, read back to be set beside each other.

Each run gets a short label for tables and figures: its model, and when two runs share a
model, the spec fields they differ in. Runs are named one by one - a topic's folder also
holds trial and offline runs, and a comparison should say what it compares.
"""
from dataclasses import dataclass
from typing import Dict, Sequence

import pandas as pd

from panel.run import ResultStore
from panel.topic import PlotStyle, Topic, load_polls

from ..saved import SavedRun, find_stem, load_run

CLAIMED = "claimed"


@dataclass(frozen=True)
class RunSet:
    """
    Attributes:
        topic: the topic all the runs answered.
        runs: label -> run, in the order they were named.
        polls: the topic's published polls, one row each, a column per option; may be empty.
        style: the topic's colours and labels for figures, or the empty style.
    """
    topic: Topic
    runs: Dict[str, SavedRun]
    polls: pd.DataFrame
    style: PlotStyle

    @property
    def claimed(self) -> Dict[str, SavedRun]:
        """
        Label -> a run whose subjects' claims stand for it: one entry, `claimed`, when every
        run asked the same subjects, else `claimed (<run>)` per distinct set of subjects.
        Empty when the dataset has no claims for the topic.
        """
        distinct = {}
        for label, run in self.runs.items():
            if run.claims:
                distinct.setdefault(frozenset(run.subjects.index), (label, run))
        if len(distinct) == 1:
            return {CLAIMED: next(iter(distinct.values()))[1]}
        return {f"{CLAIMED} ({label})": run for label, run in distinct.values()}


def labels(runs: Sequence[SavedRun]) -> Dict[str, SavedRun]:
    """
    A unique short label per run: the model, then the spec fields the runs differ in where
    the model alone does not tell them apart.
    """
    specs = [run.spec.to_dict() for run in runs]
    differing = [field for field in specs[0] if field != "model" and len({str(spec[field]) for spec in specs}) > 1]
    models = [spec["model"] for spec in specs]
    return {
        spec["model"] if models.count(spec["model"]) == 1 else f"{spec['model']} ({', '.join(f'{field}={spec[field]}' for field in differing)})": run
        for spec, run in zip(specs, runs)
    }


def load_runs(store: ResultStore, texts: Sequence[str], styled: bool = True) -> RunSet:
    """
    Read the named runs of a store's topic.

    Args:
        store: the topic's results.
        texts: per run, its stem or any text only one stem contains.
        styled: draw with the topic's `plot.json`.

    Raises:
        ValueError: on no run named, a run named twice, or whatever `find_stem` and
            `load_run` refuse.
    """
    stems = [find_stem(store, text) for text in texts]
    if not stems or len(set(stems)) != len(stems):
        raise ValueError(f"name each run to compare once; got {stems}")
    runs = [load_run(store, stem, styled) for stem in stems]
    return RunSet(topic=runs[0].topic, runs=labels(runs), polls=load_polls(runs[0].topic), style=runs[0].style)


def describe(runs: RunSet) -> str:
    """What was loaded, a line per run, and the polls."""
    width = max(len(label) for label in runs.runs)
    lines = [f"  {label:<{width}}  {len(run.subjects)} subjects, {len(run.results)} answers  {run.stem}" for label, run in runs.runs.items()]
    return "\n".join([f"{runs.topic.name}: {len(runs.runs)} runs", *lines, f"  claimed: {', '.join(runs.claimed) or 'none'}", f"  polls: {', '.join(runs.polls.index) or 'none'}"])
