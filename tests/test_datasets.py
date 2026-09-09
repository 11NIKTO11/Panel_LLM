"""Datasets declare themselves, and a profile table is checked against the shared schema going out and coming in."""
import numpy as np
import pandas as pd
import pytest

from datasets.sod.clean import own_form
from panel.analysis.aggregate import claimed_distribution
from panel.dataset import dataset_names, load_claims, load_dataset, profile_path, read_profile, write_profile
from panel.schema import load_schema
from panel.topic import load_topic

from tests.conftest import RESULTS_DIR, ROOT, require

DATASETS = dataset_names()
ATOL = 1e-12


@pytest.fixture(scope="session")
def schema():
    return load_schema()


@pytest.fixture(scope="session")
def sod():
    return load_dataset("sod")


def minimal_profile(dataset) -> pd.DataFrame:
    """Three rows carrying exactly the required columns; the values do not matter to validation."""
    return pd.DataFrame({column: ["x"] * 3 for column in dataset.required}, index=pd.Index([1, 2, 3], name="id"))


def test_schema_declares_each_column_once(schema):
    assert len(schema.columns) == sum(len(concept.columns) for concept in schema.concepts.values())


@pytest.mark.parametrize("name", DATASETS)
def test_dataset_loads(name):
    dataset = load_dataset(name)
    assert dataset.short.isascii() and "_" not in dataset.short
    for topic_name in dataset.claims:
        claims = load_claims(dataset, topic_name)
        assert set(claims.options.values()) <= set(load_topic(topic_name).options), topic_name


def test_dataset_shorts_unique():
    shorts = [load_dataset(name).short for name in DATASETS]
    assert len(set(shorts)) == len(shorts)


@pytest.mark.parametrize("name", DATASETS)
def test_profile_validates(name, schema):
    dataset = load_dataset(name)
    require(profile_path(dataset))
    assert len(read_profile(dataset, schema))


def test_required_columns(sod, schema, tmp_path):
    good = minimal_profile(sod)
    assert write_profile(sod, good, schema, tmp_path / "ok.csv").exists()
    for label, bad in {
        "missing column": good.drop(columns=[sod.required[0]]),
        "gap in a required column": good.assign(**{sod.required[0]: ["x", None, "x"]}),
        "column outside the schema": good.assign(favourite_colour="blue"),
        "duplicate id": good.set_axis(pd.Index([1, 1, 2], name="id")),
        "index not named id": good.rename_axis("row"),
    }.items():
        with pytest.raises(ValueError):
            write_profile(sod, bad, schema, tmp_path / "bad.csv")


def test_sod_profile_matches_clean_data(sod, schema):
    require(profile_path(sod))
    profile = read_profile(sod, schema)
    legacy = pd.read_csv(require(ROOT / "data" / "clean_data.csv"), index_col="id")
    assert profile.index.equals(legacy.index) and list(profile.columns) == list(legacy.columns)
    for column in legacy.columns:
        expected = legacy[column]
        if column in ("opinion_on_eu", "opinion_on_nato"):
            expected = legacy.apply(lambda row: own_form(row[column], row["gender"]), axis=1)
        same = (profile[column] == expected) | (profile[column].isna() & expected.isna())
        assert same.all(), f"{column}: {int((~same).sum())} rows differ"


def test_claimed_matches_stored(sod, schema):
    require(profile_path(sod))
    topic = load_topic("election_2021")
    stored = pd.read_csv(require(RESULTS_DIR / topic.name / "sod_claimed_votes.csv"), index_col=0).iloc[:, 0]
    stored = stored.rename(index=topic.rename_map)
    actual = claimed_distribution(read_profile(sod, schema), load_claims(sod, topic.name), topic)
    assert list(actual.index) == list(stored.index)
    np.testing.assert_allclose(actual.to_numpy(), stored.to_numpy(), rtol=0, atol=ATOL)


def test_claimed_unknown_answer_raises(sod):
    topic = load_topic("election_2021")
    frame = pd.DataFrame({"voted_party": ["an answer the map does not know"]})
    with pytest.raises(ValueError):
        claimed_distribution(frame, load_claims(sod, topic.name), topic)
