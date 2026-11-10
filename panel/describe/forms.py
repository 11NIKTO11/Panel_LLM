"""
Forms: how a dataset's values are worded inside a sentence.

The values table holds each answer as the dataset writes it. A sentence needs another form of
it - a case, agreement with the respondent's gender, a whole clause - and a closed set of
answers is cheaper to write down than to derive. A dataset's `forms.json` is that table.

    {
        "columns": {
            "gender":  {"forms": {"nom": {"Muž": "muž", "Žena": "žena"}}},
            "region":  {"forms": {"in": {"Zlínský kraj": "ve Zlínském kraji"},
                                  "of": {"Zlínský kraj": "Zlínského kraje"}}},
            "opinion": {"by": "gender", "none": ["Nevím"],
                        "forms": {"nom": {"Spokojený/á": {"Muž": "spokojený", "Žena": "spokojená"}}}},
            "jobs":    {"split": "; ", "join": " a zároveň ",
                        "forms": {"nom": {"Student": "student"}}}
        }
    }

- a column may have several named forms; a template slot picks one, `{region|in}`. The names
  are the dataset's own: a case (`nom`, `acc`), a use (`in`, `of`), a `clause`.
- a column with one form can be named without it, `{gender}`.
- a column without an entry is spliced as it stands (a number, a place name).
- `none` lists the answers that say nothing about the subject; a sentence needing one is dropped.
- `by` names the column a phrase agrees with; each phrase is then one per value of it.
- `split` / `join` are for a cell holding several answers.
- an answer that is neither in `none` nor in the form raises: the vocabulary changed, and a
  guess would reach a prompt.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple, Union

COLUMN_KEYS = {"forms", "none", "by", "split", "join"}


@dataclass(frozen=True)
class ColumnForms:
    """One column's table; see the module docstring."""
    forms: Mapping[str, Mapping[str, Any]]
    none: Tuple[str, ...] = ()
    by: Union[str, None] = None
    split: Union[str, None] = None
    join: str = ", "


@dataclass(frozen=True)
class Forms:
    """A dataset's forms, by column."""
    columns: Dict[str, ColumnForms]

    def phrase(self, column: str, value, row: Mapping, form: Union[str, None] = None) -> Union[str, None]:
        """
        The value as a sentence needs it, or None if it says nothing about the subject.

        Args:
            column: the values column.
            value: the cell.
            row: the subject's row, for a column whose phrase agrees with another.
            form: which of the column's forms; may be left out when it has exactly one.

        Raises:
            KeyError: on a form the column does not have, no form named where there are
                several, or a value - or the value of the `by` column - with no phrase.
        """
        if column not in self.columns:
            if form:
                raise KeyError(f"forms: column '{column}' has no forms, so no '{form}'")
            return str(value)
        table = self.columns[column]
        if form is None:
            if len(table.forms) != 1:
                raise KeyError(f"forms: column '{column}' has forms {sorted(table.forms)}; a slot must name one")
            form = next(iter(table.forms))
        if form not in table.forms:
            raise KeyError(f"forms: column '{column}' has no form '{form}', only {sorted(table.forms)}")
        parts = str(value).split(table.split) if table.split else [str(value)]
        phrases = [self._one(column, table, form, part, row) for part in parts]
        return None if any(phrase is None for phrase in phrases) else table.join.join(phrases)

    @staticmethod
    def _one(column: str, table: ColumnForms, form: str, value: str, row: Mapping) -> Union[str, None]:
        if value in table.none:
            return None
        phrases = table.forms[form]
        if value not in phrases:
            raise KeyError(f"forms: column '{column}', form '{form}' has no entry for {value!r}")
        phrase = phrases[value]
        if table.by is None:
            return phrase
        key = str(row[table.by])
        if key not in phrase:
            raise KeyError(f"forms: column '{column}', answer {value!r} has no phrase for {table.by}={key!r}")
        return phrase[key]


def load_forms(path: Union[str, Path]) -> Forms:
    """
    Read a dataset's forms file.

    Raises:
        FileNotFoundError: if there is no such file.
        ValueError: on an unknown key, a column without forms, forms of one column covering
            different answers, an answer both worded and listed in `none`, or a phrase whose
            shape disagrees with the column's `by`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no forms at {path}")
    columns = {}
    for column, entry in json.loads(path.read_text(encoding="utf-8"))["columns"].items():
        unknown = sorted(set(entry) - COLUMN_KEYS)
        if unknown or not entry.get("forms"):
            raise ValueError(f"{path}: column '{column}' has unknown keys {unknown} or no forms")
        table = ColumnForms(forms=entry["forms"], none=tuple(entry.get("none", ())), by=entry.get("by"),
                            split=entry.get("split"), join=entry.get("join", ", "))
        answers = {name: set(phrases) for name, phrases in table.forms.items()}
        if len({frozenset(keys) for keys in answers.values()}) != 1:
            raise ValueError(f"{path}: the forms of column '{column}' cover different answers")
        both = sorted(set(table.none) & next(iter(answers.values())))
        if both:
            raise ValueError(f"{path}: column '{column}' both words and drops {both}")
        wanted = dict if table.by else str
        bad = sorted({value for phrases in table.forms.values() for value, phrase in phrases.items()
                      if not isinstance(phrase, wanted)})
        if bad:
            raise ValueError(f"{path}: column '{column}' phrases for {bad} should be {wanted.__name__}")
        columns[column] = table
    return Forms(columns)
