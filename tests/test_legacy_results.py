"""The legacy runs are the regression fixture: their numbers must not drift through the refactor, nor through their migration to today's shape."""
import json

import numpy as np
import pytest

from panel.analysis import load_run

from tests.conftest import load_fixture, require
from tests.snapshots import aggregate_all, frame_digest, json_shape, models_by_name, reparse

AGGREGATES = load_fixture("legacy_aggregates")
REPARSE = load_fixture("legacy_reparse")
ATOL = 1e-12


def as_ids(series: dict, topic) -> dict:
    """A fixture series keyed by the native names the legacy CSVs use, rekeyed by option id."""
    return {topic.rename_map.get(column, column): value for column, value in series.items()}


@pytest.mark.parametrize("stem", sorted(AGGREGATES))
def test_aggregate_matches_fixture(stem, legacy_store, legacy_topic):
    for source, expected in AGGREGATES[stem].items():
        frame = legacy_topic.rename(legacy_store.frame(stem, normalized=source == "normalized_csv"))
        actual = aggregate_all(frame, legacy_topic)
        assert actual.keys() == expected.keys(), source
        for combination, series in expected.items():
            series = as_ids(series, legacy_topic)
            assert list(actual[combination]) == list(series), (source, combination)
            np.testing.assert_allclose(
                list(actual[combination].values()), list(series.values()),
                rtol=0, atol=ATOL, err_msg=f"{stem} {source} {combination}",
            )


@pytest.mark.parametrize("stem", sorted(stem for stem, entry in REPARSE.items() if entry["json"]))
def test_legacy_results_reparse(stem, legacy_store, legacy_topic, legacy_framing):
    expected = REPARSE[stem]
    raw = json.loads(require(legacy_store.paths(stem).results_json).read_text(encoding="utf-8"))
    model = models_by_name(legacy_topic, legacy_framing)[expected["response_model"]]
    frame = reparse(raw, model, legacy_topic.columns)
    stored = legacy_topic.rename(legacy_store.frame(stem))[legacy_topic.columns]
    assert frame.index.equals(stored.index)
    np.testing.assert_allclose(frame.to_numpy(dtype=float), stored.to_numpy(dtype=float), rtol=0, atol=ATOL)
    assert json_shape(raw) == expected["json"]["shape"]
    assert frame_digest(frame) == expected["json"]["sha256"]


def test_every_legacy_run_has_a_fixture(legacy_stems):
    assert set(legacy_stems) == set(AGGREGATES) == set(REPARSE)


@pytest.mark.parametrize("stem", sorted(AGGREGATES))
def test_a_migrated_run_is_read_by_the_analysis(stem, legacy_store):
    require(legacy_store.paths(stem).record)
    run = load_run(legacy_store, stem, styled=False)
    assert run.record["migrated"]["from"] and len(run.subjects) == run.record["n"] == REPARSE[stem]["rows"]
    assert run.claims and run.reference is not None
