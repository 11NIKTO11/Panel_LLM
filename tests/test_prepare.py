"""Prepare: a dataset's declaration, its values table, its forms, and the descriptions rendered from them."""
import importlib
import json

import pandas as pd
import pytest

from panel.dataset import INDEX, load_claims, load_dataset, read_values, validate_values
from panel.describe.forms import ColumnForms, Forms, load_forms
from panel.describe.render import read_prompts, render
from panel.describe.template import PromptTemplate, load_template
from panel.prepare import count_prompts, prepare

from tests.conftest import require

FORMS = Forms({
    "gender": ColumnForms({"nom": {"Muž": "muž", "Žena": "žena"}}),
    "region": ColumnForms({"in": {"Zlínský kraj": "ve Zlínském kraji"}, "of": {"Zlínský kraj": "Zlínského kraje"}}),
    "opinion": ColumnForms({"nom": {"Spokojený/á": {"Muž": "spokojený", "Žena": "spokojená"}}}, none=("Nevím",), by="gender"),
    "jobs": ColumnForms({"nom": {"Student": "student", "Důchodce": "důchodce"}}, split="; ", join=" a zároveň "),
})
ROW = {"gender": "Žena", "age": 28, "region": "Zlínský kraj", "opinion": "Spokojený/á", "jobs": "Student; Důchodce"}


def template(*sentences) -> PromptTemplate:
    return PromptTemplate("demo", "demo", tuple(tuple(block) for block in sentences))


@pytest.fixture(scope="session")
def sod():
    return load_dataset("sod")


def minimal_values(dataset) -> pd.DataFrame:
    return pd.DataFrame({column: ["x"] * 3 for column in dataset.columns}, index=pd.Index([1, 2, 3], name=INDEX))


def test_phrase_forms():
    assert FORMS.phrase("gender", "Žena", ROW) == "žena"
    assert FORMS.phrase("region", "Zlínský kraj", ROW, "in") == "ve Zlínském kraji"
    assert FORMS.phrase("region", "Zlínský kraj", ROW, "of") == "Zlínského kraje"
    assert FORMS.phrase("opinion", "Spokojený/á", ROW) == "spokojená"
    assert FORMS.phrase("opinion", "Spokojený/á", {**ROW, "gender": "Muž"}) == "spokojený"
    assert FORMS.phrase("opinion", "Nevím", ROW) is None
    assert FORMS.phrase("jobs", "Student; Důchodce", ROW) == "student a zároveň důchodce"
    assert FORMS.phrase("age", 28, ROW) == "28"


@pytest.mark.parametrize("column, value, form", [
    ("gender", "Jiné", None),
    ("region", "Zlínský kraj", None),
    ("region", "Zlínský kraj", "at"),
    ("age", 28, "nom"),
    ("jobs", "Student; Pilot", None),
])
def test_phrase_raises(column, value, form):
    with pytest.raises(KeyError):
        FORMS.phrase(column, value, ROW, form)


def test_phrase_unknown_agreement_raises():
    with pytest.raises(KeyError):
        FORMS.phrase("opinion", "Spokojený/á", {**ROW, "gender": "Jiné"})


@pytest.mark.parametrize("columns", [
    {"a": {"forms": {}}},
    {"a": {"forms": {"x": {"v": "p"}}, "extra": 1}},
    {"a": {"forms": {"x": {"v": "p"}, "y": {"w": "p"}}}},
    {"a": {"forms": {"x": {"v": "p"}}, "none": ["v"]}},
    {"a": {"forms": {"x": {"v": "p"}}, "by": "b"}},
    {"a": {"forms": {"x": {"v": {"m": "p"}}}}},
])
def test_load_forms_refuses(columns, tmp_path):
    path = tmp_path / "forms.json"
    path.write_text(json.dumps({"columns": columns}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_forms(path)


def test_template_renders_and_degrades():
    both = template(["Jsem {gender}, je mi {age} let."], ["Žiji {region|in}."],
                    ["Jsem {opinion} a pracuji jako {jobs}.", "Jsem {opinion}.", "Pracuji jako {jobs}."])
    assert both.columns == ("gender", "age", "region", "opinion", "jobs")
    assert both.render(ROW, FORMS) == "Jsem žena, je mi 28 let. Žiji ve Zlínském kraji. Jsem spokojená a pracuji jako student a zároveň důchodce."
    assert both.render({**ROW, "opinion": "Nevím"}, FORMS).endswith("Pracuji jako student a zároveň důchodce.")
    assert both.render({**ROW, "jobs": None}, FORMS).endswith("Jsem spokojená.")
    assert both.render({**ROW, "opinion": "Nevím", "jobs": float("nan")}, FORMS).endswith("Žiji ve Zlínském kraji.")
    assert template(["{gender} jsem."]).render(ROW, FORMS) == "Žena jsem."


def test_render_refuses(sod):
    values = pd.DataFrame([ROW], index=pd.Index([1], name=INDEX))
    with pytest.raises(ValueError):
        render(sod, template(["Mám {income}."]), values, FORMS)
    with pytest.raises(ValueError):
        render(sod, template(["Jsem {opinion}."]), values.assign(opinion="Nevím"), FORMS)


def test_dataset_declares_roles(sod):
    assert sod.short.isascii() and "_" not in sod.short
    assert set(sod.optional) | set(sod.groups) | {sod.area} <= set(sod.columns)
    assert sod.descriptions == ("default", "extended")
    assert load_claims(sod, "election_2021").column in sod.columns


def test_values_are_checked(sod):
    good = minimal_values(sod)
    validate_values(sod, good)
    validate_values(sod, good.assign(**{sod.optional[0]: [None, "x", "x"]}))
    filled = next(column for column in sod.columns if column not in sod.optional)
    for bad in (
        good.drop(columns=[filled]),
        good.assign(**{filled: ["x", None, "x"]}),
        good.assign(favourite_colour="blue"),
        good.set_axis(pd.Index([1, 1, 2], name=INDEX)),
        good.rename_axis("row"),
    ):
        with pytest.raises(ValueError):
            validate_values(sod, bad)


def test_sod_forms_and_templates_agree(sod):
    forms = load_forms(sod.forms_path)
    for name in sod.descriptions:
        used = load_template(sod.description_path(name)).columns
        assert set(used) <= set(sod.columns), name
    assert set(forms.columns) <= set(sod.columns)
    assert sorted(forms.columns["region"].forms) == ["in", "of"]


def test_sod_prompts_match_values(sod):
    require(sod.values_path)
    values, forms = read_values(sod), load_forms(sod.forms_path)
    for name in sod.descriptions:
        require(sod.prompts_path(name))
        stored = read_prompts(sod, name, values.index)[name]
        fresh = render(sod, load_template(sod.description_path(name)), values, forms)
        assert (stored == fresh).all(), name
        assert not stored.str.contains(r"[{}]|/á|\bnan\b|\bNone\b", regex=True).any(), name


def test_read_prompts_refuses_other_subjects(sod):
    require(sod.prompts_path("default"))
    with pytest.raises(ValueError):
        read_prompts(sod, "default", pd.Index([1, 2, 3], name=INDEX))


def test_count_prompts(sod):
    require(sod.prompts_path("default"))
    stored = read_prompts(sod, "default")["default"]
    count = count_prompts("sod", "default")
    assert count.prompts == len(stored) and count.distinct == stored.nunique()
    assert count.identical == stored.duplicated(keep=False).sum() and count.distinct + count.identical >= count.prompts


def test_regions_render_from_the_source():
    regions = load_dataset("regions")
    values = importlib.import_module("datasets.regions.build").build(regions)
    validate_values(regions, values)
    assert list(values.index) == list(range(1, 16)) and values["region"].iloc[0] == "Česko"
    forms = load_forms(regions.forms_path)
    citizen = render(regions, load_template(regions.description_path("citizen")), values, forms)
    direct = render(regions, load_template(regions.description_path("direct")), values, forms)
    assert citizen[1] == "Jsem obyvatel/ka Česka." and citizen[2] == "Jsem obyvatel/ka Prahy."
    assert direct[3] == "Účast a výsledek voleb do poslanecké sněmovny ve Středočeském kraji v roce 2021 budou:"
    assert citizen.nunique() == direct.nunique() == 15
    for name, fresh in (("citizen", citizen), ("direct", direct)):
        stored = read_prompts(regions, name, values.index)[name]
        assert (stored == fresh).all(), name


def test_prepare_unknown_description(sod):
    require(sod.values_path)
    with pytest.raises(ValueError):
        prepare("sod", ["nope"], rebuild_values=False)
