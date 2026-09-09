"""
The named filters English templates use, over phrase-ready values.

English needs no gender agreement, so every filter here sees only its value. Region forms come
from `forms/regions.json`, keyed by the canonical region names the reference results and the
profiles use, so an English description of a Czech region reads "in the Central Bohemian Region".
"""
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from . import grammar as en

REGION_FORMS = Path(__file__).resolve().parent / "forms" / "regions.json"


@lru_cache(maxsize=1)
def _region_forms() -> dict:
    """The area forms, read once on first use, never at import."""
    return json.loads(REGION_FORMS.read_text(encoding="utf-8"))["forms"]


def region_in(region: str) -> str:
    """
    The area as a prepositional phrase: "in the Central Bohemian Region", "in Prague".

    Raises:
        KeyError: if the area is not in forms/regions.json.
    """
    return _region_forms()[region]["in"]


def region_of(region: str) -> str:
    """
    The area as a noun phrase to follow "of": "the Central Bohemian Region", "Prague".

    Raises:
        KeyError: if the area is not in forms/regions.json.
    """
    return _region_forms()[region]["of"]


def satisfied(value) -> str:
    """"I am somewhat dissatisfied" - the answer is the predicate."""
    return f"I am {value}"


def vaccinated(value) -> str:
    """
    "I am vaccinated" / "I am not vaccinated" - a yes/no answer becomes the statement it implies.

    Raises:
        ValueError: on anything but yes or no; `usable` drops non-substantive answers first.
    """
    answer = en.parse_yes_no(str(value))
    if answer is None:
        raise ValueError(f"expected a yes/no answer, got {value!r}")
    return "I am vaccinated" if answer else "I am not vaccinated"


FILTERS = {
    "years": en.age_clause,
    "in": region_in,
    "of": region_of,
    "satisfied": satisfied,
    "vaccinated": vaccinated,
}


def usable(value) -> bool:
    """A value can fill a slot unless it is missing, empty or a non-substantive answer."""
    return not pd.isna(value) and str(value) != "" and not en.is_non_substantive(value)
