"""
Czech self-description of one synthetic grid profile.

The grid's own describer, kept apart from the survey's (`datasets/sod/data_processing_utils.py`)
because the two speak different data: a survey respondent answers ten questions, a grid
profile carries only the factors its config varies. Sharing one function meant making
every sentence in the survey describer optional, which made it worse at its own job.

What the two *do* share is the sentence wording, on purpose. The first sentence and the
politics sentence are written exactly as the survey's, and factor levels are spliced in
raw, so a grid profile built from `grid_config_sod.json` - whose levels are the survey's
own vocabulary - reads like a respondent. That is what makes grid and survey runs
comparable, and it is worth more than tidier Czech: "Příjem naší domácnosti je Do 15.000
Kč." keeps the survey's capital D rather than prettifying one side of the comparison.

`grid_config_simple.json` is *not* a prompt source - its levels are the colleague's
English ids, and this describer would produce Czenglish. It exists to read their results,
not to generate ours.
"""
import pandas as pd

from utils import czech_grammar as cz

FACTORS = ("gender", "age", "age_range", "education_level", "income_range", "interest_in_politics")


def _factor(profile, name):
    """The profile's level for one factor, or None when the grid does not vary it."""
    if name not in profile:
        return None
    value = profile[name]
    if pd.isna(value) or value == "":
        return None
    return value


def create_grid_description(profile) -> str:
    """
    Turn one grid profile into a Czech first-person description.

    Args:
        profile: a row of a generated grid, carrying some of `FACTORS`. Levels are expected
            to be survey vocabulary; `age` (an exact age) wins over `age_range` (a band).

    Returns:
        The description, or "" for a profile carrying none of the known factors.
    """
    parts = []

    facts = []
    gender = _factor(profile, "gender")
    if gender is not None:
        facts.append(f"Jsem {gender}")
    age = _factor(profile, "age")
    if age is None:
        age = _factor(profile, "age_range")
    if age is not None:
        facts.append(cz.age_clause(age))
    education = _factor(profile, "education_level")
    if education is not None:
        facts.append(f"mé vzdělání je {education}")
    if facts:
        sentence = ", ".join(facts)
        parts.append(sentence[0].upper() + sentence[1:] + ".")

    income = _factor(profile, "income_range")
    if income is not None:
        parts.append(f"Příjem naší domácnosti je {income}.")

    interest = _factor(profile, "interest_in_politics")
    if interest is not None:
        parts.append(f"{interest.capitalize()} o politiku.")

    return " ".join(parts)
