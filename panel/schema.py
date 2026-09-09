"""
The shared profile schema: concepts, and the columns that represent them.

`datasets/schema.json` is the one vocabulary every profile table is checked against, so a
column means the same thing in every dataset that carries it: `age` is always an exact age,
`age_range` always a band. A dataset emits whichever representations it has; a template
phrases whichever it knows. Common scales for grouping across datasets come in a later phase.

File shape:

    {
        "note": "...",
        "concepts": {
            "age": {"columns": {"age": "exact age in years", "age_range": "a band, as written"}},
            ...
        }
    }
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Union

from panel import paths

SCHEMA_PATH = paths.DATASETS_DIR / "schema.json"
FILE_KEYS = {"concepts"}
OPTIONAL_FILE_KEYS = {"note"}
CONCEPT_KEYS = {"columns"}


@dataclass(frozen=True)
class Schema:
    """
    Attributes:
        concepts: concept -> its columns, in file order.
        descriptions: column -> what it holds, for documentation and error messages.
    """
    concepts: Dict[str, Tuple[str, ...]]
    descriptions: Dict[str, str]

    @property
    def columns(self) -> Dict[str, str]:
        """Column -> concept, over every concept."""
        return {column: concept for concept, columns in self.concepts.items() for column in columns}


def load_schema(path: Union[str, Path] = SCHEMA_PATH) -> Schema:
    """
    Read the profile schema.

    Raises:
        FileNotFoundError: if there is no schema file.
        ValueError: on unknown keys, a concept without columns, or a column claimed twice.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no profile schema at {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    unknown = sorted(set(config) - FILE_KEYS - OPTIONAL_FILE_KEYS)
    missing = sorted(FILE_KEYS - set(config))
    if unknown or missing:
        raise ValueError(f"{path}: unknown keys {unknown}, missing keys {missing}")

    concepts: Dict[str, Tuple[str, ...]] = {}
    descriptions: Dict[str, str] = {}
    for concept, entry in config["concepts"].items():
        stray = sorted(set(entry) - CONCEPT_KEYS)
        if stray or not entry.get("columns"):
            raise ValueError(f"{path}: concept '{concept}' has unknown keys {stray} or no columns")
        for column, description in entry["columns"].items():
            if column in descriptions:
                raise ValueError(f"{path}: column '{column}' is declared by two concepts")
            descriptions[column] = description
        concepts[concept] = tuple(entry["columns"])
    return Schema(concepts=concepts, descriptions=descriptions)
