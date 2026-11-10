"""
Map the survey's raw `.sav` rows to the dataset's values.

The only code that knows the survey's raw column names (`AGE1`, `KRAJ`, `Q21`). Every answer
is kept exactly as the survey writes it - capitals, its two-gender shorthand ("spokojený/á"),
its non-answers ("Nevím") - so the values table is the survey, not a wording of it. How an
answer reads inside a sentence is `forms.json`.
"""
SEVERAL = "; "


def _employment(row) -> str | None:
    """The statuses answered yes, joined by `SEVERAL`; None when there is none."""
    statuses = [row[column] for column in row.index if "EMPLOYMENT_" in column and row[column] != "Ne"]
    return SEVERAL.join(statuses) or None


def process_respondent(row) -> dict:
    """
    One raw `.sav` row as a values row, keyed by the survey's own respondent `id`.

    Q19 is living standard, Q20 interest in politics, Q21 the 2021 vote, Q18 satisfaction with
    EU membership, Q17 with NATO membership, Q23 covid vaccination.

    Raises:
        KeyError: if the raw row lacks a column the survey is known to have.
    """
    return {
        "id": int(row["id"]),
        "gender": row["GENDER"],
        "age": int(row["AGE1"]),
        "education_level": row["EDU"],
        "region": row["KRAJ"],
        "district": row["OKRES"],
        "town_size": row["VMB"],
        "employment_status": _employment(row),
        "income_range": row["INCOMEP"],
        "living_standard": row["Q19"],
        "interest_in_politics": row["Q20"],
        "voted_party": row["Q21"],
        "opinion_on_eu": row["Q18"],
        "opinion_on_nato": row["Q17"],
        "covid_vaccinated": row["Q23"],
    }
