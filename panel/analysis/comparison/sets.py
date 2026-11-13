"""
Sets of runs worth a name, kept as data: which runs of a topic go beside each other, and
what each is called in tables and figures.

`config/comparisons.json`:

    {
        "haiku": {
            "topic": "election_2021",
            "runs": {
                "Haiku 4.5": "sod_default_cs_default_AnthropicLegacy_claude-haiku-4-5",
                "Haiku 5.5": "sod_default_cs_default_Anthropic_claude-haiku-5-5"
            }
        }
    }

A run is named by its stem or by any text only one stem of the topic contains, as on the
command line; its key is its label. The set's name is a handle for people and the name its
files are written under; a saved run is still identified by its stem.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

KEYS = {"topic", "runs"}


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


def load_comparisons(path: Union[str, Path]) -> Dict[str, Comparison]:
    """
    Every named set of the file, by name, in file order.

    Raises:
        FileNotFoundError: if there is no such file.
        ValueError: on a set with unknown or missing keys, without runs, or naming one run twice.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no comparisons at {path}")
    found = {}
    for name, entry in json.loads(path.read_text(encoding="utf-8")).items():
        if set(entry) != KEYS:
            raise ValueError(f"{path}: set '{name}' needs exactly the keys {sorted(KEYS)}, it has {sorted(entry)}")
        runs = dict(entry["runs"])
        if not runs or len(set(runs.values())) != len(runs):
            raise ValueError(f"{path}: set '{name}' needs runs, each named once")
        found[name] = Comparison(name=name, topic=entry["topic"], runs=runs)
    return found
