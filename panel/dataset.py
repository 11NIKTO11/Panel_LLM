"""
What a dataset declares, and its profile table.

`datasets/<name>/dataset.json` says where subjects come from, in which language their values
are written, which columns every row must carry and, per topic the dataset has an answer for,
where its claims map is. The profile table its `prepare` step writes lives at
`data/profiles/<name>.csv`, keyed by subject id, and is checked against the shared schema
(`panel.schema`) on the way out and on the way back in, so a bad table never reaches a run.

Config shape:

    {
        "name": "sod",                          the folder name, and the dataset's name in a run
        "short": "sod",                         stem part: ASCII, no underscore
        "title": "...",
        "language": "cs",                       the language of the values
        "source": "SoD-CZ/....sav",             what prepare reads, relative to the slice folder
        "required": ["gender", "age", ...],     columns every row must carry, non-null
        "claims": {                             topic -> the column holding the answer, and its map
            "election_2021": {"column": "voted_party", "map": "claims/election_2021.json"}
        },
        "note": "free text, ignored by code"
    }

Claims map shape - the dataset's own answer strings, sorted by what they mean for a topic:

    {
        "options": {"ANO 2011": "ANO", ...},    answer -> option short
        "abstained": ["Nebyl/a jsem volit"],    counted in turnout, in no option
        "excluded": ["Nevím", "Nechci uvést"],  non-substantive: outside every denominator
        "note": "..."
    }
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd

from panel import paths
from panel.schema import Schema, load_schema

CONFIG_NAME = "dataset.json"
REQUIRED_KEYS = {"name", "short", "title", "language", "source", "required", "claims"}
OPTIONAL_KEYS = {"note"}
CLAIM_KEYS = {"column", "map"}
MAP_KEYS = {"options", "abstained", "excluded"}
MAP_OPTIONAL_KEYS = {"note"}
INDEX = "id"


@dataclass(frozen=True)
class ClaimRef:
    """Where a dataset keeps its answer to one topic: the profile column, and the map file."""
    column: str
    map: str


@dataclass(frozen=True)
class Dataset:
    """
    Attributes:
        name: folder under `datasets/`, profile file stem, and what a run records.
        short: the dataset's part of a results stem.
        title: human label.
        language: language code of the values, e.g. "cs".
        source: what `prepare` reads, relative to `dir`.
        required: columns every profile row must carry with a value.
        claims: topic name -> where the dataset's answer to it lives.
        dir: the slice folder.
        note: free text; not used by code.
    """
    name: str
    short: str
    title: str
    language: str
    source: str
    required: Tuple[str, ...]
    claims: Dict[str, ClaimRef]
    dir: Path
    note: str = ""

    @property
    def claim_columns(self) -> Tuple[str, ...]:
        """Profile columns holding answers rather than attributes; allowed beside the schema's."""
        return tuple(sorted({ref.column for ref in self.claims.values()}))


@dataclass(frozen=True)
class Claims:
    """One dataset's answers to one topic, sorted by meaning (see the module docstring)."""
    dataset: str
    topic: str
    column: str
    options: Dict[str, str]
    abstained: Tuple[str, ...]
    excluded: Tuple[str, ...]

    @property
    def answers(self) -> Tuple[str, ...]:
        """Every answer string the map accounts for."""
        return tuple(self.options) + self.abstained + self.excluded


def dataset_names(datasets_dir: Union[str, Path] = paths.DATASETS_DIR) -> List[str]:
    """Every folder under `datasets_dir` that declares a dataset, sorted."""
    return sorted(path.parent.name for path in Path(datasets_dir).glob(f"*/{CONFIG_NAME}"))


def load_dataset(name: str, datasets_dir: Union[str, Path] = paths.DATASETS_DIR) -> Dataset:
    """
    Read one dataset declaration.

    Raises:
        FileNotFoundError: if there is no declaration of that name.
        ValueError: on unknown or missing keys, a name that disagrees with the folder, or a
            short that could not sit in a stem.
    """
    directory = Path(datasets_dir) / name
    path = directory / CONFIG_NAME
    if not path.exists():
        raise FileNotFoundError(f"no dataset declaration at {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    unknown = sorted(set(config) - REQUIRED_KEYS - OPTIONAL_KEYS)
    missing = sorted(REQUIRED_KEYS - set(config))
    if unknown or missing:
        raise ValueError(f"{path}: unknown keys {unknown}, missing keys {missing}")
    if config["name"] != name:
        raise ValueError(f"{path} declares name '{config['name']}'")
    short = config["short"]
    if not short or not short.isascii() or "_" in short:
        raise ValueError(f"{path}: short '{short}' must be non-empty ASCII without underscores")
    if not config["required"]:
        raise ValueError(f"{path}: 'required' names no columns")

    claims = {}
    for topic, entry in config["claims"].items():
        if set(entry) != CLAIM_KEYS:
            raise ValueError(f"{path}: claims for '{topic}' need exactly the keys {sorted(CLAIM_KEYS)}")
        claims[topic] = ClaimRef(column=entry["column"], map=entry["map"])

    return Dataset(
        name=config["name"], short=short, title=config["title"], language=config["language"],
        source=config["source"], required=tuple(config["required"]), claims=claims,
        dir=directory, note=config.get("note", ""),
    )


def load_claims(dataset: Dataset, topic: str) -> Claims:
    """
    The dataset's claims map for one topic.

    Raises:
        KeyError: if the dataset declares no answer for that topic.
        ValueError: on unknown or missing keys, or an answer string sorted into two groups.
    """
    if topic not in dataset.claims:
        raise KeyError(f"dataset '{dataset.name}' declares no claims for topic '{topic}'")
    ref = dataset.claims[topic]
    path = dataset.dir / ref.map
    config = json.loads(path.read_text(encoding="utf-8"))
    unknown = sorted(set(config) - MAP_KEYS - MAP_OPTIONAL_KEYS)
    missing = sorted(MAP_KEYS - set(config))
    if unknown or missing:
        raise ValueError(f"{path}: unknown keys {unknown}, missing keys {missing}")
    answers = list(config["options"]) + list(config["abstained"]) + list(config["excluded"])
    if len(set(answers)) != len(answers):
        raise ValueError(f"{path}: an answer is sorted into more than one group")
    return Claims(
        dataset=dataset.name, topic=topic, column=ref.column, options=dict(config["options"]),
        abstained=tuple(config["abstained"]), excluded=tuple(config["excluded"]),
    )


def profile_path(dataset: Dataset) -> Path:
    """Where the dataset's profile table lives once prepared."""
    return paths.PROFILES_DIR / f"{dataset.name}.csv"


def validate_profile(dataset: Dataset, frame: pd.DataFrame, schema: Schema) -> None:
    """
    Refuse a profile table a run could not trust.

    Raises:
        ValueError: if the index is not a unique `id`, a required column is missing or has
            gaps, or a column is neither in the schema nor a declared claim column.
    """
    problems = []
    if frame.index.name != INDEX:
        problems.append(f"index is named {frame.index.name!r}, not '{INDEX}'")
    if not frame.index.is_unique or frame.index.hasnans:
        problems.append("index has duplicates or gaps")
    missing = [column for column in dataset.required if column not in frame.columns]
    if missing:
        problems.append(f"required columns missing: {missing}")
    gaps = {column: int(frame[column].isna().sum()) for column in dataset.required if column in frame.columns}
    gaps = {column: count for column, count in gaps.items() if count}
    if gaps:
        problems.append(f"required columns with missing values: {gaps}")
    allowed = set(schema.columns) | set(dataset.claim_columns)
    unknown = [column for column in frame.columns if column not in allowed]
    if unknown:
        problems.append(f"columns outside the schema: {unknown}")
    if problems:
        raise ValueError(f"profile '{dataset.name}': " + "; ".join(problems))


def write_profile(
    dataset: Dataset, frame: pd.DataFrame, schema: Schema, path: Union[str, Path, None] = None
) -> Path:
    """
    Validate and store a dataset's profile table, at `data/profiles/<name>.csv` by default.

    Raises:
        ValueError: see `validate_profile`.
    """
    validate_profile(dataset, frame, schema)
    path = Path(path) if path else profile_path(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=True, encoding="utf-8")
    return path


def read_profile(
    dataset: Dataset, schema: Union[Schema, None] = None, path: Union[str, Path, None] = None
) -> pd.DataFrame:
    """
    A dataset's stored profile table, validated on the way in.

    Raises:
        FileNotFoundError: if the dataset has not been prepared.
        ValueError: see `validate_profile`.
    """
    path = Path(path) if path else profile_path(dataset)
    if not path.exists():
        raise FileNotFoundError(f"no profile table at {path}; run the dataset's prepare step first")
    frame = pd.read_csv(path, index_col=INDEX, encoding="utf-8")
    validate_profile(dataset, frame, schema or load_schema())
    return frame
