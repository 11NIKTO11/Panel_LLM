"""
Turn survey answers into Czech prompt phrases, and render the description templates.

Composition root for the survey's half of prompt building: it owns the filters that carry
Czech grammar over survey answers - including the region case forms - and hands them to the
generic renderer in `panel/describe/template.py`. It reads the profile table `prepare` wrote,
so a description never describes rows that are not there.

Usage:
    python -m datasets.sod.describe [--variant self_description_v1]
"""
import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Union

import pandas as pd

from panel.describe.lang.cs import grammar as cz
from panel.dataset import load_dataset, read_profile
from panel.describe.template import RowFilter, load_template, render_frame, write_table

from .clean import is_non_substantive

SLICE_DIR = Path(__file__).resolve().parent      # datasets/sod - templates and forms
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


def _satisfied(value) -> str:
    """"jsem spíše nespokojená" - cleaning already resolved the answer to the respondent's gender."""
    return f"jsem {value}"


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
    "satisfied": _satisfied,
    "vaccinated": RowFilter(_vaccinated),
}


def usable(value) -> bool:
    """A survey answer can fill a slot unless it is missing, empty or non-substantive."""
    return not pd.isna(value) and str(value) != "" and not is_non_substantive(value)


def render(people: pd.DataFrame, variant: str = DEFAULT_VARIANT) -> pd.Series:
    """
    Describe every respondent, as the named variant.

    Args:
        people: the survey's profile table, indexed by survey id.
        variant: template file stem under `datasets/sod/`; also the column and table name.

    Returns:
        One description per row, named for the variant.
    """
    template = load_template(TEMPLATE_DIR / f"{variant}.json")
    return render_frame(template, people, FILTERS, usable)


def main(argv: Union[list, None] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--variant", default=DEFAULT_VARIANT, help="template file stem")
    args = parser.parse_args(argv)

    people = read_profile(load_dataset("sod"))
    descriptions = render(people, args.variant)
    path = write_table(descriptions)
    print(f"{len(descriptions)} descriptions -> {path}")
    print(descriptions.iloc[0])


if __name__ == "__main__":
    main()
