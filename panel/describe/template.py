"""
Description templates: a prompt as data, rendered over a frame.

A template is a list of **sentences**, each a list of **alternatives in priority order**, and
the first alternative whose slots are all filled wins. That is how a sentence degrades when an
answer is missing, and how a thinner profile drops a sentence entirely. Templates live in
`config/descriptions/`, one file per variant, and are shared across datasets: they are written
against the schema's column ids, not against one dataset's columns.

Template shape:

    {
        "name": "self_v1",              the file stem, the table name and the column
        "short": "v1",                  its part of a results stem: ASCII, no underscore
        "language": "cs",               which language pack supplies the filters
        "standalone": false,            true when the description is the whole prompt
        "note": "...",
        "sentences": [
            ["Jsem {gender}, {age|years}, mé vzdělání je {education_level}.",
             "Jsem {gender}, {age_range|years}, mé vzdělání je {education_level}."],
            ["Z hlediska zaměstnání jsem {employment_status} a příjem je {income_range}.",
             "Z hlediska zaměstnání jsem {employment_status}.",
             "Příjem naší domácnosti je {income_range}."]
        ]
    }

- `{column}` splices the value as it stands; `{column|filter}` passes it through a named
  filter the language pack supplies. Filters are where grammar lives, because Czech case and
  agreement cannot be expressed in a template.
- a filter wrapped in `RowFilter` receives `(value, row)` instead of just the value, because
  agreement can depend on another column: an adjective about the respondent has to know
  their gender, which is not the column being spliced.
- the renderer capitalises each sentence, so filters return lowercase phrases.
- a value that is absent, empty, or rejected by the caller's `usable` predicate leaves its
  slot unfilled, which drops that alternative.
- a `standalone` description is the whole prompt: a run appends no question after it.

Rendering over a dataset, with its language pack, is `panel.describe.render`; storing the
result is `panel.describe.tables`.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Tuple, Union

import pandas as pd

from panel import paths

SLOT = re.compile(r"\{([a-z_0-9]+)(?:\|([a-z_]+))?\}")
REQUIRED_KEYS = {"name", "short", "language", "sentences"}
OPTIONAL_KEYS = {"note", "standalone"}


def _default_usable(value) -> bool:
    """Absent and empty answers cannot fill a slot."""
    return not pd.isna(value) and str(value) != ""


@dataclass(frozen=True)
class RowFilter:
    """
    A filter that needs the whole row, not only the cell it is spliced into.

    Marked rather than sniffed, so a language pack's filter table says which filters read
    elsewhere in the row.
    """
    fn: Callable

    def __call__(self, value, row):
        return self.fn(value, row)


@dataclass(frozen=True)
class PromptTemplate:
    """
    One description variant.

    Attributes:
        name: names the file, the table and its column.
        short: the description's part of a results stem.
        language: the language pack whose filters render it.
        sentences: alternatives in priority order, per sentence.
        standalone: the description is the whole prompt; no question follows it.
        note: free text; not used by code.
    """
    name: str
    short: str
    language: str
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

    def filters(self) -> Tuple[str, ...]:
        """Every filter name the template asks for."""
        return tuple({name for block in self.sentences for line in block
                      for _, name in SLOT.findall(line) if name})

    def render(
        self,
        row: pd.Series,
        filters: Union[Mapping[str, Callable], None] = None,
        usable: Union[Callable, None] = None,
    ) -> str:
        """
        The description of one row: every sentence whose slots this row can fill.

        Raises:
            KeyError: if the template names a filter the caller did not supply.
        """
        filters = filters or {}
        usable = usable or _default_usable
        missing = [name for name in self.filters() if name not in filters]
        if missing:
            raise KeyError(f"template '{self.name}' needs filters {missing}")

        sentences = []
        for block in self.sentences:
            for line in block:
                text = self._fill(line, row, filters, usable)
                if text is not None:
                    sentences.append(text[0].upper() + text[1:] if text else text)
                    break
        return " ".join(sentences)

    def _fill(self, line, row, filters, usable) -> Union[str, None]:
        """The line with its slots filled, or None if this row cannot fill one of them."""
        out, position = [], 0
        for match in SLOT.finditer(line):
            column, filter_name = match.group(1), match.group(2)
            if column not in row.index:
                return None
            value = row[column]
            if not usable(value):
                return None
            out.append(line[position:match.start()])
            out.append(str(self._apply(filters.get(filter_name), value, row)))
            position = match.end()
        out.append(line[position:])
        return "".join(out)

    @staticmethod
    def _apply(filter_fn, value, row):
        """The value as the slot's filter renders it; a `RowFilter` also sees the row."""
        if isinstance(filter_fn, RowFilter):
            return filter_fn(value, row)
        return filter_fn(value) if filter_fn else value


def render_frame(
    template: PromptTemplate,
    frame: pd.DataFrame,
    filters: Union[Mapping[str, Callable], None] = None,
    usable: Union[Callable, None] = None,
) -> pd.Series:
    """One description per row, named for the template so it lands as that column."""
    return frame.apply(lambda row: template.render(row, filters, usable), axis=1).rename(template.name)


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
        name=config["name"], short=short, language=config["language"], sentences=sentences,
        standalone=bool(config.get("standalone", False)), note=config.get("note", ""),
    )


def load_description(name: str, descriptions_dir: Union[str, Path] = paths.DESCRIPTIONS_DIR) -> PromptTemplate:
    """The description of that name from `config/descriptions/`; see `load_template` for what it raises."""
    return load_template(Path(descriptions_dir) / f"{name}.json")
