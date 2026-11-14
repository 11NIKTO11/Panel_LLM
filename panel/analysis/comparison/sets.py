"""
Sets of runs worth a name, kept as data: which runs of a topic go beside each other, and
what each is called in tables and figures.

`config/topics/<topic>/comparisons.json`, optional:

    {
        "haiku": {
            "Haiku 4.5": "sod_default_cs_default_AnthropicLegacy_claude-haiku-4-5",
            "Haiku 5.5": "sod_default_cs_default_Anthropic_claude-haiku-5-5"
        }
    }

A set is `label -> run`. A run is named by its stem or by any text only one stem of the topic
contains, as on the command line; its key is its label. The set's name is a handle for people
and the name its files are written under; a saved run is still identified by its stem.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

from panel import paths

CONFIG_NAME = "comparisons.json"


@dataclass(frozen=True)
class Comparison:
    """
    Attributes:
        name: the set's name.
        topic: the topic whose runs it holds.
        runs: label -> a stem or a text only one stem contains, in file order.
    """
    name: str
    topic: str
    runs: Dict[str, str]


def load_comparisons(topic: str, topics_dir: Union[str, Path] = paths.TOPICS_DIR) -> Dict[str, Comparison]:
    """
    Every named set of a topic, by name, in file order; none if it has no `comparisons.json`.

    Raises:
        ValueError: on a set that is not a mapping of labels to runs, without runs, or naming
            one run twice.
    """
    path = Path(topics_dir) / topic / CONFIG_NAME
    if not path.exists():
        return {}
    found = {}
    for name, runs in json.loads(path.read_text(encoding="utf-8")).items():
        if not isinstance(runs, dict) or not all(isinstance(run, str) for run in runs.values()):
            raise ValueError(f"{path}: set '{name}' must map labels to runs")
        if not runs or len(set(runs.values())) != len(runs):
            raise ValueError(f"{path}: set '{name}' needs runs, each named once")
        found[name] = Comparison(name=name, topic=topic, runs=dict(runs))
    return found
