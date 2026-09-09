"""
Turn survey answers into Czech prompt phrases, and render the description templates.

Composition root for the survey's half of prompt building: it owns the filters that carry
Czech grammar over survey answers - including the region case forms - and hands them to the
generic renderer in `panel/describe/template.py`. Run it from the repo root, or call `render()` from
`prepare.ipynb` right after `clean_data.csv` is written - doing it in the same pass is what
keeps a table from lagging the data it describes.

Usage:
    python -m datasets.sod.code.describe [--variant self_description_v1]
"""
import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Union

import pandas as pd

from panel.describe.lang.cs import grammar as cz
from panel.describe.template import RowFilter, load_template, render_frame, write_table

from .clean import is_non_substantive_responses

SLICE_DIR = Path(__file__).resolve().parents[1]      # datasets/sod - templates and forms
TEMPLATE_DIR = SLICE_DIR
DEFAULT_VARIANT = "self_description_v1"
REGION_FORMS = SLICE_DIR / "forms" / "regions.json"


@lru_cache(maxsize=1)
def _region_forms() -> dict:
    """The region case forms, read once on first use (never at import)."""
    return json.loads(REGION_FORMS.read_text(encoding="utf-8"))["forms"]


def region_in(region_name: str) -> str:
    """
    The region as a prepositional phrase: "ve Středočeském kraji", "v Praze".

    The preposition is part of the form because it vocalises with what follows, which is
    what the rules this replaced could not see.

    Raises:
        KeyError: if the region is not in datasets/sod/forms/regions.json.
    """
    return _region_forms()[region_name]["in"]


def region_of(region_name: str) -> str:
    """
    The region in the genitive: "Středočeského kraje", "Prahy".

    Raises:
        KeyError: if the region is not in datasets/sod/forms/regions.json.
    """
    return _region_forms()[region_name]["of"]


def _living_standard(value) -> str:
    """"mám spíše dobrou" / "nemám ani dobrou, ani špatnou" - the verb follows the answer."""
    verb = cz.declension_negation_ne("mám") if "ani" in str(value) else "mám"
    return f"{verb} {value}"


def _gender(row) -> cz.Gender:
    """The respondent's grammatical gender, from the survey's own "muž" / "žena"."""
    return cz.Gender.FEMALE if str(row["gender"]).lower() == "žena" else cz.Gender.MALE


def _own_form(value: str, gender: cz.Gender) -> str:
    """
    One respondent's form of an answer the survey writes for both genders.

    "rozhodně spokojený/á" -> "rozhodně spokojený" or "rozhodně spokojená". The slash is the
    survey's shorthand, not something a person would say, so it never reaches a prompt.

    Raises:
        ValueError: if the answer carries no slash form, which means the vocabulary changed.
    """
    masculine, slash, feminine = str(value).rpartition("/")
    if not slash:
        raise ValueError(f"expected a '<masculine>/<feminine ending>' answer, got {value!r}")
    return masculine[:-1] + feminine if gender is cz.Gender.FEMALE else masculine


def _satisfied(value, row) -> str:
    """"jsem spíše nespokojená" - the answer is an adjective, so it agrees with the gender."""
    return f"jsem {_own_form(value, _gender(row))}"


def _vaccinated(value, row) -> str:
    """
    "jsem očkovaný" / "nejsem očkovaná" - a yes/no answer becomes the verb it implies.

    Raises:
        ValueError: on anything but yes or no. `usable` drops "nevím" and "nechci uvést"
            before a slot is filled, so reaching here with one is a bug, not an answer.
    """
    vaccinated = cz.parse_yes_no(str(value))
    if vaccinated is None:
        raise ValueError(f"expected an ano/ne answer, got {value!r}")
    verb = "jsem" if vaccinated else cz.declension_negation_ne("jsem")
    return f"{verb} {cz.declension_gender_ya('očkovan', _gender(row))}"


FILTERS = {
    "years": cz.age_clause,
    "in": region_in,
    "of": region_of,
    "have": _living_standard,
    "satisfied": RowFilter(_satisfied),
    "vaccinated": RowFilter(_vaccinated),
}


def usable(value) -> bool:
    """A survey answer can fill a slot unless it is missing, empty or non-substantive."""
    return not pd.isna(value) and str(value) != "" and not is_non_substantive_responses(value)


def render(people: pd.DataFrame, variant: str = DEFAULT_VARIANT) -> pd.Series:
    """
    Describe every respondent, as the named variant.

    Args:
        people: cleaned respondents, indexed by survey id.
        variant: template file stem under `datasets/sod/`; also the column and table name.

    Returns:
        One description per row, named for the variant.
    """
    template = load_template(TEMPLATE_DIR / f"{variant}.json")
    return render_frame(template, people, FILTERS, usable)


def main(argv: Union[list, None] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--variant", default=DEFAULT_VARIANT, help="template file stem")
    parser.add_argument("--data", default="data/clean_data.csv", help="cleaned respondents")
    args = parser.parse_args(argv)

    people = pd.read_csv(args.data, index_col=0)
    descriptions = render(people, args.variant)
    path = write_table(descriptions)
    print(f"{len(descriptions)} descriptions -> {path}")
    print(descriptions.iloc[0])


if __name__ == "__main__":
    main()
