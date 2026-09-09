"""
A topic: what a run is about, as data.

The vocabulary a run answers with, the question it is asked, and the system prompt that
frames the task all live in `config/topics/<name>.json`. This module only reads them -
`RunSpec.topic` holds the name, so a saved run reproduces its own topic from its record.

Config shape:

    {
        "name": "election_2021",
        "title": "...",
        "instructions": "<system prompt>",     null for a topic whose runs sent none
        "question": " ...",                    appended to every description
        "reference_results": "data/....csv",   null when there is nothing to compare to
        "turnout": {                           null for a topic with no attendance dimension
            "yes": {"column": "Voted",     "aliases": ["p_voted"]},
            "no":  {"column": "Not Voted"}
        },
        "options": {
            "ANO 2011": {"short": "ANO", "aliases": ["p_ANO"]},
            ...
        },
        "note": "free text, ignored by code"
    }

Each option is named once; `short` defaults to the name and `aliases` to none. **The order
of the option keys is the order the model sees them** - it becomes the `enum` order in the
schema sent to the provider, so it is prompt content, not presentation. `json` preserves
file order, so the file is the record of it; reordering the keys changes the prompt.

`yes` / `no` are roles, not labels: the engine treats them asymmetrically (aggregation
weights by the `yes` branch), so it has to know which is which whatever the columns are
called. The response models keep the field names `voted_or_not.voted` / `.not_voted`
regardless, because saved result JSON is keyed by them.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd

from panel import paths

REQUIRED_KEYS = {"name", "title", "instructions", "question", "reference_results", "turnout", "options"}
OPTIONAL_KEYS = {"note"}
ENTRY_KEYS = {"short", "aliases"}
TURNOUT_ROLES = ("yes", "no")


@dataclass(frozen=True)
class Topic:
    """
    One topic, loaded from its config file.

    Attributes:
        name: the config file's stem, the results folder, and `RunSpec.topic`.
        title: human label, for figures and reports.
        instructions: system prompt framing the task; None for a topic whose runs sent none.
        question: appended to every respondent description by default.
        options: what the model may answer, in the order it sees them.
        short: option name -> short label, for plots and tables; every option has one.
        aliases: foreign column name -> our column name, covering options and turnout, for
            reading someone else's results.
        turnout: (yes column, no column) for a topic with an attendance dimension, else None.
        reference_results: CSV of real-world results to compare against, or None.
        note: free text about the topic's state; not used by code.
    """
    name: str
    title: str
    instructions: Union[str, None]
    question: str
    options: Tuple[str, ...]
    short: Dict[str, str]
    aliases: Dict[str, str]
    turnout: Union[Tuple[str, str], None]
    reference_results: Union[str, None]
    note: str = ""

    @property
    def by_short(self) -> Dict[str, str]:
        """Short label -> option name, for reading a label back to a column."""
        return {label: option for option, label in self.short.items()}

    @property
    def columns(self) -> List[str]:
        """Result columns in canonical order: attendance first, if any, then the options."""
        return list(self.turnout or ()) + list(self.options)

    @property
    def normalize_groups(self) -> List[List[str]]:
        """Column groups that each have to sum to 1 per row, for `ResultStore.save`."""
        groups = [list(self.turnout)] if self.turnout else []
        groups.append(list(self.options))
        return groups

    def rename(self, frame: pd.DataFrame) -> pd.DataFrame:
        """A foreign results frame with its columns renamed to ours, via `aliases`."""
        return frame.rename(columns=self.aliases)

    def load_reference(self) -> pd.DataFrame:
        """
        The topic's real-world results.

        Raises:
            ValueError: if the topic declares no reference file, or the file does not carry
                every column the topic declares.
            FileNotFoundError: if the declared file is missing.
        """
        if not self.reference_results:
            raise ValueError(f"topic '{self.name}' declares no reference_results")
        frame = pd.read_csv(paths.ROOT / self.reference_results, encoding="utf-8-sig")
        missing = [column for column in self.columns if column not in frame.columns]
        if missing:
            raise ValueError(f"{self.reference_results} is missing {len(missing)} topic columns: {missing}")
        return frame


def _claim(aliases: Dict[str, str], foreign: str, ours: str, path: Path) -> None:
    """Record one alias, refusing a name that is already taken."""
    if foreign in aliases:
        raise ValueError(f"{path}: alias '{foreign}' is claimed by both '{aliases[foreign]}' and '{ours}'")
    aliases[foreign] = ours


def load_topic(name: str, topics_dir: Union[str, Path] = paths.TOPICS_DIR) -> Topic:
    """
    Read one topic config. Option order is the order in the file (see the module docstring).

    Args:
        name: file stem under `topics_dir`, and the topic's own name.
        topics_dir: where the configs live; the repo's `config/topics` by default.

    Raises:
        FileNotFoundError: if there is no config of that name.
        ValueError: if the config has unknown or missing keys, names a different topic, or
            reuses a short label, an alias or a column name.
    """
    path = Path(topics_dir) / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"no topic config at {path}")

    config = json.loads(path.read_text(encoding="utf-8"))
    unknown = sorted(set(config) - REQUIRED_KEYS - OPTIONAL_KEYS)
    missing = sorted(REQUIRED_KEYS - set(config))
    if unknown or missing:
        raise ValueError(f"{path}: unknown keys {unknown}, missing keys {missing}")
    if config["name"] != name:
        raise ValueError(f"{path} declares name '{config['name']}'")

    options: Tuple[str, ...] = tuple(config["options"])
    if not options:
        raise ValueError(f"{path}: no options")

    turnout = None
    aliases: Dict[str, str] = {}
    if config["turnout"]:
        columns = []
        for role in TURNOUT_ROLES:
            entry = config["turnout"][role]
            columns.append(entry["column"])
            for foreign in entry.get("aliases", ()):
                _claim(aliases, foreign, entry["column"], path)
        turnout = (columns[0], columns[1])

    short: Dict[str, str] = {}
    for option, entry in config["options"].items():
        stray = sorted(set(entry) - ENTRY_KEYS)
        if stray:
            raise ValueError(f"{path}: option '{option}' has unknown keys {stray}")
        short[option] = entry.get("short", option)
        for foreign in entry.get("aliases", ()):
            _claim(aliases, foreign, option, path)

    labels = list(short.values())
    if len(set(labels)) != len(labels):
        raise ValueError(f"{path}: short labels are not unique: {sorted(labels)}")
    clash = sorted(set(aliases) & set(list(options) + list(turnout or ())))
    if clash:
        raise ValueError(f"{path}: aliases collide with our own column names: {clash}")

    return Topic(
        name=config["name"],
        title=config["title"],
        instructions=config["instructions"],
        question=config["question"],
        options=options,
        short=short,
        aliases=aliases,
        turnout=turnout,
        reference_results=config["reference_results"],
        note=config.get("note", ""),
    )
