"""
A topic's vocabulary: what a run may answer with, as data.

`config/topics/<name>/topic.json` holds the language-neutral half of a topic: the option ids,
each with its native name and foreign aliases, the turnout roles, and where the reference
results are. Everything the model reads - system prompt, question, labels, schema text - is a
framing (`panel.topic.framing`), one file per language beside it. `RunSpec.topic` holds the
name, so a saved run reproduces its own vocabulary from its record.

Config shape:

    {
        "name": "election_2021",
        "reference": {                          null when there is nothing to compare to
            "path": "data/election_data.csv",   relative to the repo root
            "key": "region",                    the column naming each row
            "total": "Česko"                    the key of the row that is the whole
        },
        "turnout": {                            null for a topic with no attendance dimension
            "yes": {"column": "Voted", "aliases": ["p_voted"]},
            "no":  {"column": "Not Voted"}
        },
        "options": {
            "ANO": {"name": "ANO 2011", "aliases": ["p_ANO"]},
            ...
        },
        "note": "free text, ignored by code"
    }

- **option ids are the keys**: pure ASCII, and the columns of every result table, so runs in
  two languages on one topic stack. `name` is the official label, which a framing shows unless
  it overrides it; `aliases` are foreign column names, for reading someone else's results.
- **the order of the option keys is the order the model sees them** - it becomes the `enum`
  order in the schema sent to the provider, so it is prompt content, not presentation. `json`
  preserves file order, so the file is the record of it; reordering the keys changes the prompt.
- `yes` / `no` are roles, not labels: aggregation weights by the `yes` branch, so the engine
  has to know which is which whatever the columns are called. The response models keep the
  field names `voted_or_not.voted` / `.not_voted` regardless, because saved JSON is keyed by them.
- `rename(frame)` turns native names and aliases into ids, so a legacy CSV, the reference CSV
  and a colleague's `p_*` table all read as ids without a file on disk changing.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd

from panel import paths

CONFIG_NAME = "topic.json"
REQUIRED_KEYS = {"name", "reference", "turnout", "options"}
OPTIONAL_KEYS = {"note"}
OPTION_KEYS = {"name", "aliases"}
REFERENCE_KEYS = {"path", "key", "total"}
TURNOUT_ROLES = ("yes", "no")


@dataclass(frozen=True)
class Reference:
    """Where a topic's real-world results are, which column keys the rows, and which row is the whole."""
    path: str
    key: str
    total: str


@dataclass(frozen=True)
class Topic:
    """
    Attributes:
        name: the folder under `config/topics/`, the results folder, and `RunSpec.topic`.
        options: option ids, in the order the model sees them; also the result columns.
        names: option id -> its native name, the label a framing shows by default.
        aliases: foreign column name -> our column, over options and turnout.
        turnout: (yes column, no column) for a topic with an attendance dimension, else None.
        reference: the real-world results to compare against, or None.
        note: free text; not used by code.
    """
    name: str
    options: Tuple[str, ...]
    names: Dict[str, str]
    aliases: Dict[str, str]
    turnout: Union[Tuple[str, str], None]
    reference: Union[Reference, None]
    note: str = ""

    @property
    def columns(self) -> List[str]:
        """Result columns in canonical order: attendance first, if any, then the option ids."""
        return list(self.turnout or ()) + list(self.options)

    @property
    def normalize_groups(self) -> List[List[str]]:
        """Column groups that each have to sum to 1 per row, for `ResultStore.save`."""
        groups = [list(self.turnout)] if self.turnout else []
        groups.append(list(self.options))
        return groups

    @property
    def rename_map(self) -> Dict[str, str]:
        """Native name or alias -> our column, what `rename` applies."""
        return {**{name: option for option, name in self.names.items()}, **self.aliases}

    def rename(self, frame: pd.DataFrame) -> pd.DataFrame:
        """A results frame with native names and aliases turned into our columns."""
        return frame.rename(columns=self.rename_map)

    def load_reference(self) -> pd.DataFrame:
        """
        The topic's real-world results in our columns, one row per key, the total row included.

        Raises:
            ValueError: if the topic declares no reference, or the file lacks the key column,
                a topic column, or the total row.
            FileNotFoundError: if the declared file is missing.
        """
        if not self.reference:
            raise ValueError(f"topic '{self.name}' declares no reference results")
        frame = self.rename(pd.read_csv(paths.ROOT / self.reference.path, encoding="utf-8-sig"))
        missing = [column for column in [self.reference.key] + self.columns if column not in frame.columns]
        if missing:
            raise ValueError(f"{self.reference.path} is missing columns {missing}")
        if self.reference.total not in set(frame[self.reference.key]):
            raise ValueError(f"{self.reference.path} has no row keyed '{self.reference.total}'")
        return frame


def topic_names(topics_dir: Union[str, Path] = paths.TOPICS_DIR) -> List[str]:
    """Every folder under `topics_dir` that declares a topic, sorted."""
    return sorted(path.parent.name for path in Path(topics_dir).glob(f"*/{CONFIG_NAME}"))


def _claim(taken: Dict[str, str], foreign: str, ours: str, path: Path) -> None:
    """Record one alias, refusing a name that is already taken."""
    if foreign in taken:
        raise ValueError(f"{path}: '{foreign}' is claimed by both '{taken[foreign]}' and '{ours}'")
    taken[foreign] = ours


def load_topic(name: str, topics_dir: Union[str, Path] = paths.TOPICS_DIR) -> Topic:
    """
    Read one topic's vocabulary. Option order is the order in the file (see the module docstring).

    Raises:
        FileNotFoundError: if there is no topic of that name.
        ValueError: on unknown or missing keys, a name that disagrees with the folder, an id
            that is not ASCII, a name or alias claimed twice or colliding with our own columns.
    """
    path = Path(topics_dir) / name / CONFIG_NAME
    if not path.exists():
        raise FileNotFoundError(f"no topic at {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    unknown = sorted(set(config) - REQUIRED_KEYS - OPTIONAL_KEYS)
    missing = sorted(REQUIRED_KEYS - set(config))
    if unknown or missing:
        raise ValueError(f"{path}: unknown keys {unknown}, missing keys {missing}")
    if config["name"] != name:
        raise ValueError(f"{path} declares name '{config['name']}'")

    options = tuple(config["options"])
    if not options:
        raise ValueError(f"{path}: no options")
    for option in options:
        if not option or not option.isascii():
            raise ValueError(f"{path}: option id {option!r} must be non-empty ASCII")

    turnout = None
    taken: Dict[str, str] = {}
    if config["turnout"]:
        columns = []
        for role in TURNOUT_ROLES:
            entry = config["turnout"][role]
            columns.append(entry["column"])
            for foreign in entry.get("aliases", ()):
                _claim(taken, foreign, entry["column"], path)
        turnout = (columns[0], columns[1])

    names: Dict[str, str] = {}
    for option, entry in config["options"].items():
        stray = sorted(set(entry) - OPTION_KEYS)
        if stray or "name" not in entry:
            raise ValueError(f"{path}: option '{option}' has unknown keys {stray} or no name")
        names[option] = entry["name"]
        for foreign in entry.get("aliases", ()):
            _claim(taken, foreign, option, path)
    if len(set(names.values())) != len(names):
        raise ValueError(f"{path}: option names are not unique")

    ours = set(options) | set(turnout or ())
    clash = sorted(set(taken) & ours)
    clash += sorted(label for option, label in names.items() if label in ours and label != option)
    if clash:
        raise ValueError(f"{path}: names or aliases collide with our own columns: {clash}")

    reference = None
    if config["reference"]:
        if set(config["reference"]) != REFERENCE_KEYS:
            raise ValueError(f"{path}: reference needs exactly the keys {sorted(REFERENCE_KEYS)}")
        reference = Reference(**config["reference"])

    return Topic(
        name=config["name"], options=options, names=names, aliases=taken, turnout=turnout,
        reference=reference, note=config.get("note", ""),
    )
