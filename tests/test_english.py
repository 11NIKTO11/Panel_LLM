"""The English axis, offline: the pack, the template, the framing's schema, and the English edition of the survey."""
import numpy as np
import pandas as pd
import pytest

from panel.analysis.aggregate import claimed_distribution
from panel.dataset import load_claims, load_dataset, profile_path, read_profile
from panel.describe.lang import load_language
from panel.describe.lang.en import grammar as en
from panel.describe.render import render
from panel.describe.template import load_description
from panel.run import ResultStore, RunSpec, execute, resolve
from panel.run.fake import FakeClient
from panel.topic import load_framing, load_topic, make_response_models

from tests.conftest import require

EN = load_language("en")
SELF_V1_EN = load_description("self_v1_en")

PERSON = {
    "gender": "man", "age": 28, "age_range": "25-34 years old", "education_level": "higher education",
    "region": "Středočeský kraj", "district": "Nymburk", "town_size": "less than 1,000 inhabitants",
    "employment_status": "full-time employee", "income_range": "30,001 - 40,000 CZK",
    "living_standard": "somewhat good", "interest_in_politics": "very interested",
}
GRID_ROW = {"gender": "woman", "education_level": "secondary education with GCSE/maturita", "age": 37,
            "income_range": "up to 15,000 CZK", "interest_in_politics": "rather interested"}


def describe(row: dict) -> str:
    return SELF_V1_EN.render(pd.Series(row), EN.filters, EN.usable)


def test_english_render():
    assert SELF_V1_EN.language == "en" and SELF_V1_EN.short == "v1-en"
    assert describe(PERSON) == (
        "I am a man, aged 28, with higher education. I live in the Central Bohemian Region, in the Nymburk district, "
        "in a municipality of less than 1,000 inhabitants. My employment status is full-time employee and my household "
        "income is 30,001 - 40,000 CZK. My standard of living is somewhat good. I am very interested in politics."
    )
    assert describe({**PERSON, "income_range": "Don't know, I don't want to say"}).count("My employment status is full-time employee.") == 1
    assert describe(GRID_ROW) == (
        "I am a woman, aged 37, with secondary education with GCSE/maturita. My household income is up to 15,000 CZK. "
        "I am rather interested in politics."
    )
    assert describe({**{k: v for k, v in GRID_ROW.items() if k != "age"}, "age_range": "60+"}).startswith("I am a woman, aged 60 or older,")


@pytest.mark.parametrize(("age", "clause"), [
    (28, "aged 28"), ("30-44", "aged between 30 and 44"), ("25-34 years old", "aged between 25 and 34"),
    ("60+", "aged 60 or older"), ("More than 60 years old", "aged more than 60"),
])
def test_english_age_clause(age, clause):
    assert en.age_clause(age) == clause


def test_english_framing_builds_a_schema():
    topic = load_topic("election_2021")
    framing = load_framing(topic, "en")
    probabilities, pick = make_response_models(topic, framing)
    enum = probabilities.model_json_schema()["$defs"]["OptionProbability"]["properties"]["name"]["enum"]
    assert enum[0] == "ANO 2011" and "Other party" in enum and len(enum) == len(topic.options)
    assert "took part" in probabilities.model_json_schema()["$defs"]["TurnoutProbability"]["description"]
    row = probabilities(voted_or_not={"voted": 0.6, "not_voted": 0.4}, parties=[{"name": "Other party", "probability": 1.0}]).to_series()
    assert row["Other"] == 1.0 and list(row.index) == topic.columns


def test_english_edition_matches_czech():
    sod, sod_en = load_dataset("sod"), load_dataset("sod_en")
    require(profile_path(sod))
    require(profile_path(sod_en))
    czech, english = read_profile(sod), read_profile(sod_en)
    assert english.index.equals(czech.index) and list(english.columns) == list(czech.columns)
    for column in ("age", "region", "district"):
        assert (english[column] == czech[column]).all(), column
    assert english["income_range"].isna().sum() == czech["income_range"].isna().sum()
    assert english["employment_status"].isna().sum() == czech["employment_status"].isna().sum()
    assert set(english["gender"]) == {"man", "woman"}
    topic = load_topic("election_2021")
    np.testing.assert_allclose(
        claimed_distribution(english, load_claims(sod_en, topic.name), topic).to_numpy(),
        claimed_distribution(czech, load_claims(sod, topic.name), topic).to_numpy(), rtol=0, atol=1e-12,
    )


def test_english_run_offline(tmp_path):
    spec = RunSpec(dataset="sod_en", description="self_v1_en", topic="election_2021", framing="en",
                   client="Fake", model="fake-model", sample=5)
    require(profile_path(load_dataset("sod_en")))
    run = resolve(spec)
    assert run.stem == "sod-en_v1-en_en_Fake_fake-model_VProb_t=0.0_subset=5"
    prompt = run.prompt(run.rows.iloc[0])
    assert prompt.endswith("I voted for:") and prompt.startswith("I am a ")
    execute(run, ResultStore(tmp_path, spec.topic), FakeClient(), max_workers=2)
    assert len(ResultStore(tmp_path, spec.topic).frame(run.stem)) == 5
    with pytest.raises(ValueError):
        resolve(spec.variant(framing="cs"))
    with pytest.raises(ValueError):
        resolve(spec.variant(dataset="sod"))
