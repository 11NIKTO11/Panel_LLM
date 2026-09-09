"""
Map the English edition of the survey's raw `.sav` rows to the profile schema.

The same rows, ids and columns as the Czech edition; the value labels are its translations,
with two quirks of their own - the gender label "Man," carries a stray comma, and the smallest
settlement band lacks its unit - and no two-gender shorthand to resolve. Values come out
phrase-ready in English: first letter lowered where a label would sit mid-sentence, proper
nouns and currency codes untouched. Region names come back as the canonical (Czech) names
every reference and language pack keys on, via `regions.json` beside this module.
"""
import json
from functools import lru_cache
from pathlib import Path

from panel.describe.lang.en import grammar as en

REGIONS = Path(__file__).resolve().parent / "regions.json"


@lru_cache(maxsize=1)
def _regions() -> dict:
    """English label -> canonical region name, read once on first use."""
    return json.loads(REGIONS.read_text(encoding="utf-8"))["regions"]


def region(label: str) -> str:
    """
    Raises:
        KeyError: if the label is not one the English edition uses.
    """
    return _regions()[label]


def _employment(row) -> str | None:
    """The statuses answered yes, comma-joined and lower-cased; None when there is none."""
    columns = [column for column in row.index if "EMPLOYMENT_" in column]
    statuses = [row[column] for column in columns if row[column] != "No"]
    return ", ".join(statuses).lower() or None


def _income(value) -> str | None:
    return None if not value or en.is_non_substantive(value) else en.decapitalize(value)


def _town_size(value: str) -> str:
    if "Less than 1,000" in value:
        value += " inhabitants"
    return en.decapitalize(value)


def process_respondent(row) -> dict:
    """
    One raw `.sav` row as a profile row; see the Czech edition for what each raw column is.

    Raises:
        KeyError: if the raw row lacks a column the survey is known to have, or names a region
            the English edition does not use.
    """
    return {
        "id": int(row["id"]),
        "gender": row["GENDER"].rstrip(",").lower(),
        "age": int(row["AGE1"]),
        "age_range": row["AGE2"],
        "education_level": en.decapitalize(row["EDU"]),
        "region": region(row["KRAJ"]),
        "district": row["OKRES"],
        "town_size": _town_size(row["VMB"]),
        "employment_status": _employment(row),
        "income_range": _income(row["INCOMEP"]),
        "living_standard": row["Q19"].lower(),
        "interest_in_politics": row["Q20"].lower(),
        "voted_party": row["Q21"],
        "opinion_on_eu": row["Q18"].lower(),
        "opinion_on_nato": row["Q17"].lower(),
        "covid_vaccinated": row["Q23"].lower(),
    }
