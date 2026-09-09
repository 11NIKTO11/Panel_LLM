"""Descriptions by example: what the shared templates and the Czech pack produce for one row, with no data on disk."""
import pandas as pd
import pytest

from panel.dataset import load_dataset
from panel.describe.lang import load_language
from panel.describe.lang.cs import grammar as cz
from panel.describe.render import render
from panel.describe.template import PromptTemplate, load_description

CS = load_language("cs")
SELF_V1 = load_description("self_v1")
SELF_V2 = load_description("self_v2")

SOD = {
    "gender": "muž", "age": 28, "age_range": "25–34 let", "education_level": "vysokoškolské vzdělání",
    "region": "Středočeský kraj", "district": "Nymburk", "town_size": "méně než 1.000 obyvatel",
    "employment_status": "zaměstnanec na plný úvazek", "income_range": "30.001 - 40.000 Kč",
    "living_standard": "spíše dobrou", "interest_in_politics": "velmi se zajímám",
    "opinion_on_eu": "spíše spokojený", "opinion_on_nato": "rozhodně nespokojený", "covid_vaccinated": "ano",
}
GRID = {
    "gender": "žena", "education_level": "středoškolské vzdělání s maturitou", "age": 37,
    "income_range": "Do 15.000 Kč", "interest_in_politics": "spíše se zajímám",
}

PERSON = "Jsem muž, je mi 28 let, mé vzdělání je vysokoškolské vzdělání."
PLACE = "Žiji ve Středočeském kraji, v okresu Nymburk a obci o velikosti méně než 1.000 obyvatel."
WORK_AND_INCOME = "Z hlediska zaměstnání jsem zaměstnanec na plný úvazek a příjem naší domácnosti je 30.001 - 40.000 Kč."
WORK = "Z hlediska zaměstnání jsem zaměstnanec na plný úvazek."
INCOME = "Příjem naší domácnosti je 30.001 - 40.000 Kč."
STANDARD = "Mám spíše dobrou životní úroveň."
INTEREST = "Velmi se zajímám o politiku."
EU = "Jsem spíše spokojený s tím, že je Česká republika členem Evropské unie."
NATO = "Jsem rozhodně nespokojený s tím, že je Česká republika členem NATO."
COVID = "Jsem očkovaný proti covidu-19."
GRID_PERSON = "Jsem žena, je mi 37 let, mé vzdělání je středoškolské vzdělání s maturitou."
GRID_REST = "Příjem naší domácnosti je Do 15.000 Kč. Spíše se zajímám o politiku."


def describe(template: PromptTemplate, row: dict) -> str:
    return template.render(pd.Series(row), CS.filters, CS.usable)


def test_complete_profile():
    assert describe(SELF_V1, SOD) == " ".join([PERSON, PLACE, WORK_AND_INCOME, STANDARD, INTEREST])


@pytest.mark.parametrize(("changes", "expected"), [
    ({"income_range": float("nan")}, [PERSON, PLACE, WORK, STANDARD, INTEREST]),
    ({"employment_status": None}, [PERSON, PLACE, INCOME, STANDARD, INTEREST]),
    ({"employment_status": None, "income_range": None}, [PERSON, PLACE, STANDARD, INTEREST]),
    ({"living_standard": "nevím"}, [PERSON, PLACE, WORK_AND_INCOME, INTEREST]),
    ({"interest_in_politics": "nevím"}, [PERSON, PLACE, WORK_AND_INCOME, STANDARD]),
    ({"living_standard": "ani dobrou, ani špatnou"}, [PERSON, PLACE, WORK_AND_INCOME, "Nemám ani dobrou, ani špatnou životní úroveň.", INTEREST]),
], ids=["no income", "no employment", "neither", "standard nevim", "interest nevim", "negated standard"])
def test_degradation(changes, expected):
    assert describe(SELF_V1, {**SOD, **changes}) == " ".join(expected)


@pytest.mark.parametrize(("region", "phrase"), [
    ("Středočeský kraj", "ve Středočeském kraji"),
    ("Zlínský kraj", "ve Zlínském kraji"),
    ("Hlavní město Praha", "v Praze"),
    ("Kraj Vysočina", "v Kraji Vysočina"),
])
def test_region_forms_carry_the_preposition(region, phrase):
    assert f"Žiji {phrase}, v okresu" in describe(SELF_V1, {**SOD, "region": region})


def test_unknown_region_raises():
    with pytest.raises(KeyError):
        describe(SELF_V1, {**SOD, "region": "Neexistující kraj"})


@pytest.mark.parametrize(("age", "clause"), [
    (28, "je mi 28 let"),
    ("18-29", "je mi mezi 18 a 29 lety"),
    ("25–34 let", "je mi mezi 25 a 34 lety"),
    ("60+", "je mi 60 a více let"),
    ("Více než 60 let", "je mi více než 60 let"),
    ("do 24", "je mi do 24 let"),
])
def test_age_clause(age, clause):
    assert cz.age_clause(age) == clause


def test_v2_adds_three_sentences():
    assert describe(SELF_V2, SOD) == " ".join([PERSON, PLACE, WORK_AND_INCOME, STANDARD, INTEREST, EU, NATO, COVID])


def test_v2_agrees_with_gender_and_drops_non_answers():
    row = {**SOD, "gender": "žena", "opinion_on_eu": "spíše spokojená", "opinion_on_nato": "nevím", "covid_vaccinated": "ne"}
    assert describe(SELF_V2, row) == " ".join([
        "Jsem žena, je mi 28 let, mé vzdělání je vysokoškolské vzdělání.", PLACE, WORK_AND_INCOME, STANDARD, INTEREST,
        "Jsem spíše spokojená s tím, že je Česká republika členem Evropské unie.",
        "Nejsem očkovaná proti covidu-19.",
    ])
    assert describe(SELF_V2, {**SOD, "covid_vaccinated": "nechci uvést"}).endswith(NATO)


@pytest.mark.parametrize(("row", "person"), [
    (GRID, GRID_PERSON),
    ({**{k: v for k, v in GRID.items() if k != "age"}, "age_range": "30-44"}, "Jsem žena, je mi mezi 30 a 44 lety, mé vzdělání je středoškolské vzdělání s maturitou."),
    ({**GRID, "age_range": "60+"}, GRID_PERSON),
], ids=["exact age", "banded age", "both: exact wins"])
def test_shared_template_over_a_grid_row(row, person):
    assert describe(SELF_V1, row) == f"{person} {GRID_REST}"


def test_row_that_fills_no_sentence():
    row = {"gender": "muž", "age_range": "45-59"}
    assert describe(SELF_V1, row) == ""
    with pytest.raises(ValueError):
        render(load_dataset("grid"), SELF_V1, pd.DataFrame([row], index=pd.Index([1], name="id")))


def test_language_mismatch_raises():
    english = PromptTemplate(name="x", short="x", language="en", sentences=(("I am {gender}.",),))
    with pytest.raises(ValueError):
        render(load_dataset("sod"), english, pd.DataFrame([SOD], index=pd.Index([1], name="id")))


def test_unknown_language_raises():
    with pytest.raises(ValueError):
        load_language("xx")
