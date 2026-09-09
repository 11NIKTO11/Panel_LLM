"""
The named filters Czech templates use, over phrase-ready values.

A `{column|filter}` slot hands its value to one of these; a `RowFilter` also sees the row,
because gender agreement needs a column other than the one being spliced. Region forms come
from `forms/regions.json`, the whole prepositional phrase per area, because Czech vocalises
the preposition (v -> ve) and a closed set of fifteen is cheaper to tabulate than to derive.
"""
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from panel.describe.template import RowFilter

from . import grammar as cz

REGION_FORMS = Path(__file__).resolve().parent / "forms" / "regions.json"


@lru_cache(maxsize=1)
def _region_forms() -> dict:
    """The area case forms, read once on first use, never at import."""
    return json.loads(REGION_FORMS.read_text(encoding="utf-8"))["forms"]


def region_in(region: str) -> str:
    """
    The area as a prepositional phrase: "ve Středočeském kraji", "v Praze".

    Raises:
        KeyError: if the area is not in forms/regions.json.
    """
    return _region_forms()[region]["in"]


def region_of(region: str) -> str:
    """
    The area in the genitive: "Středočeského kraje", "Prahy".

    Raises:
        KeyError: if the area is not in forms/regions.json.
    """
    return _region_forms()[region]["of"]


def have(value) -> str:
    """"mám spíše dobrou" / "nemám ani dobrou, ani špatnou" - the verb follows a negated answer."""
    verb = cz.declension_negation_ne("mám") if "ani" in str(value) else "mám"
    return f"{verb} {value}"


def satisfied(value) -> str:
    """"jsem spíše nespokojená" - the answer already agrees with the gender, resolved at cleaning."""
    return f"jsem {value}"


def _gender(row) -> cz.Gender:
    """The row's grammatical gender, from the Czech "muž" / "žena" in its gender column."""
    return cz.Gender.FEMALE if str(row["gender"]).lower() == cz.Gender.FEMALE.value else cz.Gender.MALE


def vaccinated(value, row) -> str:
    """
    "jsem očkovaný" / "nejsem očkovaná" - a yes/no answer becomes the verb it implies.

    Raises:
        ValueError: on anything but yes or no; `usable` drops non-substantive answers before
            a slot is filled, so reaching here with one is a bug, not an answer.
    """
    answer = cz.parse_yes_no(str(value))
    if answer is None:
        raise ValueError(f"expected an ano/ne answer, got {value!r}")
    verb = "jsem" if answer else cz.declension_negation_ne("jsem")
    return f"{verb} {cz.declension_gender_ya('očkovan', _gender(row))}"


FILTERS = {
    "years": cz.age_clause,
    "in": region_in,
    "of": region_of,
    "have": have,
    "satisfied": satisfied,
    "vaccinated": RowFilter(vaccinated),
}


def usable(value) -> bool:
    """A value can fill a slot unless it is missing, empty or a non-substantive answer."""
    return not pd.isna(value) and str(value) != "" and not cz.is_non_substantive(value)
