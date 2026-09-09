"""The analysis core: grouping, comparison and common levels, held to the tables the former notebook wrote."""
import numpy as np
import pandas as pd
import pytest

from panel import paths
from panel.analysis.aggregate import aggregate_results_from_df
from panel.analysis.compare import METRICS, distance_table, distances, pairwise, smooth
from panel.analysis.groups import by, claimed_by, predicted_by, spread_by, to_levels
from panel.dataset import load_claims, load_dataset, profile_path, read_profile
from panel.schema import load_schema

from tests.conftest import RESULTS_DIR, ROOT, require

PAIRS = [("Predicted", "Actual"), ("Predicted", "Claimed"), ("Actual", "Claimed")]
STORED_MAE = sorted(path.name[len("regional_pairwise_mae_"):-4] for path in (RESULTS_DIR / "election_2021_legacy" / "analysis").glob("regional_pairwise_mae_sod_*.csv")) if RESULTS_DIR.exists() else []
STORED_COV = sorted(path.name[len("cov_matrix_"):-4] for path in (RESULTS_DIR / "election_2021_legacy" / "analysis").glob("cov_matrix_sod_*VProb*.csv")) if RESULTS_DIR.exists() else []


@pytest.fixture(scope="session")
def legacy_people():
    return pd.read_csv(require(ROOT / "data" / "legacy_clean_data.csv"), index_col="id")


@pytest.mark.parametrize("stem", STORED_MAE)
def test_regional_mae_matches_stored(stem, legacy_store, legacy_topic, legacy_people):
    topic = legacy_topic
    results = topic.rename(legacy_store.frame(stem, normalized=True))
    claims = load_claims(load_dataset("sod"), topic.name)
    total = topic.reference.total
    sources = {
        "Predicted": predicted_by(results, legacy_people, "region", topic, total=total),
        "Claimed": claimed_by(legacy_people, "region", claims, topic, total=total),
        "Actual": topic.load_reference().set_index("region"),
    }
    sources = {name: frame.loc[sources["Actual"].index] for name, frame in sources.items()}
    table = pairwise(sources, topic.options, pairs=PAIRS, decimals=3)
    stored = pd.read_csv(legacy_store.analysis(f"regional_pairwise_mae_{stem}.csv"), index_col="region")
    pd.testing.assert_frame_equal(table, stored, check_names=False)


@pytest.mark.parametrize("stem", STORED_COV)
def test_covariance_matches_stored(stem, legacy_store, legacy_topic):
    results = legacy_topic.rename(legacy_store.frame(stem, normalized=True))
    stored = legacy_topic.rename(pd.read_csv(legacy_store.analysis(f"cov_matrix_{stem}.csv"), index_col=0))
    stored.index = [legacy_topic.rename_map.get(name, name) for name in stored.index]
    covariance = results[list(legacy_topic.options)].cov()
    np.testing.assert_allclose(covariance.loc[stored.index, stored.columns].to_numpy(), stored.to_numpy(), rtol=0, atol=1e-12)


def test_by_total_is_the_aggregate_of_everything(legacy_store, legacy_topic, legacy_people):
    results = legacy_topic.rename(legacy_store.frame(legacy_store.runs()[0]))
    table = predicted_by(results, legacy_people, "region", legacy_topic, total="all")
    assert table.index.name == "region" and "all" in table.index and len(table) == 15
    pd.testing.assert_series_equal(table.loc["all"], aggregate_results_from_df(results, legacy_topic), check_names=False)
    assert abs(table[list(legacy_topic.options)].sum(axis=1) - 1).max() < 1e-9


def test_by_rejects_unknown_key():
    frame = pd.DataFrame({"x": [1, 2]}, index=pd.Index([1, 2], name="id"))
    with pytest.raises(KeyError):
        by(frame, "region", lambda part: part.sum())


def test_distances_of_identical_distributions_are_zero():
    p = [0.5, 0.3, 0.2]
    result = distances(p, p)
    assert set(result) == set(METRICS)
    assert result["MAE"] == 0 and result["RMSE"] == 0 and result["Wasserstein Distance"] == 0
    assert abs(result["KL Divergence"]) < 1e-9 and abs(result["Jensen-Shannon Divergence"]) < 1e-9
    assert result["Correlation"] == pytest.approx(1.0)


def test_distances_grow_with_moved_mass():
    near = distances([0.5, 0.3, 0.2], [0.45, 0.35, 0.2])
    far = distances([0.5, 0.3, 0.2], [0.1, 0.1, 0.8])
    assert all(far[metric] > near[metric] for metric in METRICS if metric != "Correlation")
    with pytest.raises(ValueError):
        distances([0.5, 0.5], [0.0, 0.0])


def test_distance_table_and_smooth():
    reference = pd.DataFrame({"a": [0.6, 0.2], "b": [0.4, 0.8]}, index=["g1", "g2"])
    other = pd.DataFrame({"a": [0.6, 0.0], "b": [0.4, 1.0]}, index=["g1", "g2"])
    table = distance_table(reference, other, ["a", "b"])
    assert list(table.columns) == list(METRICS) and table.loc["g1", "MAE"] == 0 and table.loc["g2", "MAE"] == pytest.approx(0.2)
    smoothed = smooth(other, ["a", "b"])
    assert (smoothed[["a", "b"]] > 0).all().all() and smoothed[["a", "b"]].sum(axis=1).round(9).eq(1).all()
    with pytest.raises(ValueError):
        pairwise({"x": reference, "y": other}, ["a", "b"], metric="Nonsense")


def test_spread_by_shape(legacy_store, legacy_topic, legacy_people):
    results = legacy_topic.rename(legacy_store.frame(legacy_store.runs()[0])).join(legacy_people[["region"]])
    spread = spread_by(results, "region", legacy_topic.options)
    assert list(spread.index) == list(legacy_topic.options) and spread.shape[1] == 14 and (spread >= 0).all().all()


def test_to_levels_age():
    schema = load_schema()
    sod = load_dataset("sod")
    require(profile_path(sod))
    people = read_profile(sod, schema)
    levels = to_levels(people, "age", schema)
    assert set(levels) <= set(schema.concepts["age"].levels) and levels.name == "age"
    assert (levels[people["age"] < 30] == "18-29").all() and (levels[people["age"] >= 60] == "60+").all()
    banded = pd.DataFrame({"age_range": ["30-44", "60+"]}, index=pd.Index([1, 2], name="id"))
    assert list(to_levels(banded, "age", schema)) == ["30-44", "60+"]
    with pytest.raises(ValueError):
        to_levels(people[["age_range"]], "age", schema)
    with pytest.raises(ValueError):
        to_levels(people, "income", schema)
