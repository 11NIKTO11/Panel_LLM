"""
Description templates: a prompt as data, rendered over a dataset's values.

A template belongs to one dataset and lives in its folder, `datasets/<name>/descriptions/`,
because it names that dataset's columns. It is a list of **sentences**, each a list of
**alternatives in priority order**; the first alternative whose slots can all be filled wins,
which is how a sentence degrades when an answer is missing, and disappears when none fits.

    {
        "name": "self_v1",          the file stem, the prompt table's name and its column
        "short": "v1",              its part of a results stem: ASCII, no underscore
        "standalone": false,        true when the description is the whole prompt
        "note": "...",
        "sentences": [
            ["Jsem {gender}, je mi {age} let, mé vzdělání je {education_level}."],
            ["Z hlediska zaměstnání jsem {employment_status} a příjem je {income_range}.",
             "Z hlediska zaměstnání jsem {employment_status}.",
             "Příjem naší domácnosti je {income_range}."]
        ]
    }

- `{column}` splices that column's value as the dataset's forms word it (`panel.describe.forms`);
  `{column|form}` picks one of several forms, e.g. `{region|in}` and `{region|of}`.
- a slot stays unfilled when the value is absent or is one of the column's non-answers; that
  drops the alternative.
- the renderer capitalises each sentence, so forms are lowercase phrases.
- a variant is additive: new wording is a new file, never an edit of one a run has recorded.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Tuple, Union

import pandas as pd

from .forms import Forms

SLOT = re.compile(r"\{([a-z_0-9]+)(?:\|([a-z_0-9]+))?\}")
REQUIRED_KEYS = {"name", "short", "sentences"}
OPTIONAL_KEYS = {"note", "standalone"}


@dataclass(frozen=True)
class PromptTemplate:
    """
    One description variant.

    Attributes:
        name: names the file, the prompt table and its column.
        short: the description's part of a results stem.
        sentences: alternatives in priority order, per sentence.
        standalone: the description is the whole prompt; no question follows it.
        note: free text; not used by code.
    """
    name: str
    short: str
    sentences: Tuple[Tuple[str, ...], ...]
    standalone: bool = False
    note: str = ""

    @property
    def columns(self) -> Tuple[str, ...]:
        """Every column the template reads, in order of first appearance."""
        seen = {}
        for block in self.sentences:
            for line in block:
                for column, _ in SLOT.findall(line):
                    seen.setdefault(column, None)
        return tuple(seen)

    def render(self, row: Mapping, forms: Forms) -> str:
        """The description of one row: every sentence this row can fill, in order."""
        sentences = []
        for block in self.sentences:
            for line in block:
                text = self._fill(line, row, forms)
                if text is not None:
                    sentences.append(text[:1].upper() + text[1:])
                    break
        return " ".join(sentences)

    @staticmethod
    def _fill(line: str, row: Mapping, forms: Forms) -> Union[str, None]:
        """The line with its slots filled, or None if this row cannot fill one of them."""
        out, position = [], 0
        for match in SLOT.finditer(line):
            column, form = match.group(1), match.group(2)
            value = row[column]
            if pd.isna(value) or str(value) == "":
                return None
            phrase = forms.phrase(column, value, row, form)
            if phrase is None:
                return None
            out += [line[position:match.start()], phrase]
            position = match.end()
        return "".join(out + [line[position:]])


def load_template(path: Union[str, Path]) -> PromptTemplate:
    """
    Read a template file. Its `name` must match its filename, so the variant has one name.

    Raises:
        FileNotFoundError: if there is no such file.
        ValueError: on unknown or missing keys, a name that disagrees with the filename, a
            short that could not sit in a stem, or a sentence without alternatives.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no description template at {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    unknown = sorted(set(config) - REQUIRED_KEYS - OPTIONAL_KEYS)
    missing = sorted(REQUIRED_KEYS - set(config))
    if unknown or missing:
        raise ValueError(f"{path}: unknown keys {unknown}, missing keys {missing}")
    if config["name"] != path.stem:
        raise ValueError(f"{path} declares name '{config['name']}'")
    short = config["short"]
    if not short or not short.isascii() or "_" in short:
        raise ValueError(f"{path}: short '{short}' must be non-empty ASCII without underscores")
    sentences = tuple(tuple(block) for block in config["sentences"])
    if not sentences or any(not block for block in sentences):
        raise ValueError(f"{path}: every sentence needs at least one alternative")
    return PromptTemplate(
        name=config["name"], short=short, sentences=sentences,
        standalone=bool(config.get("standalone", False)), note=config.get("note", ""),
    )
