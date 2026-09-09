"""
Prompt tables: render a description template over a frame, and read the result back.

A description is data, not a runtime callable. A dataset renders its template once, at
cleanup, into `data/prompts/<variant>.csv`; a run then names that variant and reads the
column. So the prompt a run used is a stored artifact rather than the output of whatever
the describer happened to be that day, and `RunSpec` stays serialisable.

Template shape - one JSON file per variant, named for the variant:

    {
        "name": "self_description_v1",
        "sentences": [
            ["Jsem {gender}, {age|years}, mé vzdělání je {education_level}."],
            ["Z hlediska zaměstnání jsem {employment_status} a příjem je {income_range}.",
             "Z hlediska zaměstnání jsem {employment_status}.",
             "Příjem naší domácnosti je {income_range}."]
        ]
    }

- one entry per sentence; the entry's lines are **alternatives in priority order**, and the
  first whose slots are all filled wins. That is how a sentence degrades when an answer is
  missing, and how a thinner profile drops a sentence entirely.
- `{column}` splices the value as it stands; `{column|filter}` passes it through a named
  filter the caller supplies. Filters are where grammar lives, because Czech case and
  agreement cannot be expressed in a template.
- a filter wrapped in `RowFilter` receives `(value, row)` instead of just the value, because
  agreement can depend on another column: an adjective about the respondent has to know
  their gender, which is not the column being spliced.
- the renderer capitalises each sentence, so filters return lowercase phrases.
- a value that is absent, empty, or rejected by the caller's `usable` predicate leaves its
  slot unfilled, which drops that alternative.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Mapping, Sequence, Tuple, Union

import pandas as pd

TABLE_DIR = "data/prompts"
SLOT = re.compile(r"\{([a-z_0-9]+)(?:\|([a-z_]+))?\}")


def _default_usable(value) -> bool:
    """Absent and empty answers cannot fill a slot."""
    return not pd.isna(value) and str(value) != ""


@dataclass(frozen=True)
class RowFilter:
    """
    A filter that needs the whole row, not only the cell it is spliced into.

    Marked rather than sniffed, so a template's filter table says which filters read
    elsewhere in the row. `_satisfied` in the survey's describer is the case that forced it:
    "rozhodně spokojený/á" has to agree with `gender`.
    """
    fn: Callable

    def __call__(self, value, row):
        return self.fn(value, row)


@dataclass(frozen=True)
class PromptTemplate:
    """One description variant. `name` names the file, the table and its column."""
    name: str
    sentences: Tuple[Tuple[str, ...], ...]
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


def load_template(path: Union[str, Path]) -> PromptTemplate:
    """
    Read a template file. Its `name` must match its filename, so the variant has one name.

    Raises:
        ValueError: if the name disagrees with the filename, or a sentence has no lines.
    """
    path = Path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["name"] != path.stem:
        raise ValueError(f"{path} declares name '{config['name']}'")
    sentences = tuple(tuple(block) for block in config["sentences"])
    if not sentences or any(not block for block in sentences):
        raise ValueError(f"{path}: every sentence needs at least one alternative")
    return PromptTemplate(name=config["name"], sentences=sentences, note=config.get("note", ""))


def render_frame(
    template: PromptTemplate,
    frame: pd.DataFrame,
    filters: Union[Mapping[str, Callable], None] = None,
    usable: Union[Callable, None] = None,
) -> pd.Series:
    """One description per row, named for the template so it lands as that column."""
    return frame.apply(lambda row: template.render(row, filters, usable), axis=1).rename(template.name)


def write_table(descriptions: pd.Series, table_dir: Union[str, Path] = TABLE_DIR) -> Path:
    """
    Store a rendered variant as `<table_dir>/<name>.csv`, keyed by the frame's index.

    A variant is additive: rendering a *different* wording means a new name, never an
    overwrite, so a run's recorded variant keeps meaning what it meant.
    """
    directory = Path(table_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{descriptions.name}.csv"
    descriptions.to_frame().to_csv(path, index=True)
    return path


def read_table(
    name: str,
    expected_index: Union[pd.Index, None] = None,
    table_dir: Union[str, Path] = TABLE_DIR,
) -> pd.DataFrame:
    """
    A rendered variant, as a one-column frame to join onto the data.

    Args:
        name: the variant, which is also the file stem and the column name.
        expected_index: the index of the rows about to be run. Given it, coverage is checked
            here rather than discovered as a NaN prompt in the middle of a paid run.

    Raises:
        FileNotFoundError: if that variant has not been rendered.
        ValueError: if the table does not cover exactly `expected_index`, or repeats a key.
    """
    path = Path(table_dir) / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"no prompt table at {path}; render it from its template first")

    table = pd.read_csv(path, index_col=0)
    if not table.index.is_unique:
        raise ValueError(f"{path}: index has duplicates")
    if expected_index is not None:
        missing = expected_index.difference(table.index)
        extra = table.index.difference(expected_index)
        if len(missing) or len(extra):
            raise ValueError(
                f"{path} does not match the data: {len(missing)} rows without a description, "
                f"{len(extra)} descriptions without a row. Re-render the variant."
            )
        table = table.loc[expected_index]
    return table
