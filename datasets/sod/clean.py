"""
Map the survey's raw `.sav` rows to the profile schema.

Survey-specific by design: it knows the raw column names (`AGE1`, `KRAJ`, `Q21`) and the
answer strings they carry, and changes when the survey's own columns do. Values come out
phrase-ready in Czech - lower-cased where the survey capitalises a mid-sentence word, and
with its two-gender shorthand ("spokojený/á") resolved against the respondent's gender - so
no template or filter downstream has to know the survey's encoding. Every answer is emitted;
which of them reach a prompt is a template's decision.
"""
from panel.describe.lang.cs import grammar as cz

FEMININE = "žena"
MASCULINE = "muž"


def own_form(value: str, gender: str) -> str:
    """
    One respondent's form of an answer the survey writes for both genders.

    "rozhodně spokojený/á" -> "rozhodně spokojený" or "rozhodně spokojená". A non-substantive
    answer has no slash and passes through unchanged.

    Raises:
        ValueError: on a slash-less answer that is not non-substantive, which means the
            vocabulary changed, or on a gender the survey does not use.
    """
    masculine, slash, feminine = value.rpartition("/")
    if not slash:
        if cz.is_non_substantive(value):
            return value
        raise ValueError(f"expected a '<masculine>/<feminine ending>' answer, got {value!r}")
    if gender == FEMININE:
        return masculine[:-1] + feminine
    if gender == MASCULINE:
        return masculine
    raise ValueError(f"unknown gender {gender!r}")


def _employment(row) -> str | None:
    """The statuses answered yes, comma-joined and lower-cased; None when there is none."""
    columns = [column for column in row.index if "EMPLOYMENT_" in column]
    statuses = [row[column] for column in columns if row[column] != "Ne"]
    return ", ".join(statuses).lower() or None


def _income(value) -> str | None:
    return None if not value or cz.is_non_substantive(value) else cz.decapitalize(value)


def _town_size(value: str) -> str:
    if "Méně než 1.000" in value:
        value += " obyvatel"
    return cz.decapitalize(value)


def process_respondent(row) -> dict:
    """
    One raw `.sav` row as a profile row.

    `id` is the survey's own respondent identifier, so descriptions and results join on
    identity rather than row position. `age` is the exact age (AGE1), `age_range` the survey's
    own band (AGE2, "25–34 let"). Q18 asks about EU membership, Q17 about NATO, Q23 about
    covid vaccination.

    Raises:
        KeyError: if the raw row lacks a column the survey is known to have.
        ValueError: see `own_form`.
    """
    gender = row["GENDER"].lower()
    return {
        "id": int(row["id"]),
        "gender": gender,
        "age": int(row["AGE1"]),
        "age_range": row["AGE2"],
        "education_level": row["EDU"].lower(),
        "region": row["KRAJ"],
        "district": row["OKRES"],
        "town_size": _town_size(row["VMB"]),
        "employment_status": _employment(row),
        "income_range": _income(row["INCOMEP"]),
        "living_standard": row["Q19"].lower(),
        "interest_in_politics": row["Q20"].lower(),
        "voted_party": row["Q21"],
        "opinion_on_eu": own_form(row["Q18"].lower(), gender),
        "opinion_on_nato": own_form(row["Q17"].lower(), gender),
        "covid_vaccinated": row["Q23"].lower(),
    }
