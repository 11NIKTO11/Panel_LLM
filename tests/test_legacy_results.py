"""The legacy runs are the regression fixture: their numbers must not drift through the refactor."""
import json

import numpy as np
import pytest

from tests.conftest import load_fixture, require
from tests.snapshots import aggregate_all, frame_digest, json_shape, models_by_name, reparse

AGGREGATES = load_fixture("legacy_aggregates")
REPARSE = load_fixture("legacy_reparse")
ATOL = 1e-12


@pytest.mark.parametrize("stem", sorted(AGGREGATES))
def test_aggregate_matches_fixture(stem, legacy_store, legacy_topic):
    for source, expected in AGGREGATES[stem].items():
        frame = legacy_store.frame(stem, normalized=source == "normalized_csv")
        actual = aggregate_all(frame, legacy_topic)
        assert actual.keys() == expected.keys(), source
        for combination, series in expected.items():
            assert list(actual[combination]) == list(series), (source, combination)
            np.testing.assert_allclose(
                list(actual[combination].values()), list(series.values()),
                rtol=0, atol=ATOL, err_msg=f"{stem} {source} {combination}",
            )


@pytest.mark.parametrize("stem", sorted(stem for stem, entry in REPARSE.items() if entry["json"]))
def test_legacy_results_reparse(stem, legacy_store, legacy_topic):
    expected = REPARSE[stem]
    raw = json.loads(require(legacy_store.paths(stem).results_json).read_text(encoding="utf-8"))
    frame = reparse(raw, models_by_name(legacy_topic)[expected["response_model"]], legacy_topic.columns)
    stored = legacy_store.frame(stem)[legacy_topic.columns]
    assert frame.index.equals(stored.index)
    np.testing.assert_allclose(frame.to_numpy(dtype=float), stored.to_numpy(dtype=float), rtol=0, atol=ATOL)
    assert json_shape(raw) == expected["json"]["shape"]
    assert frame_digest(frame) == expected["json"]["sha256"]


def test_every_legacy_run_has_a_fixture(legacy_store):
    assert set(legacy_store.runs()) == set(AGGREGATES) == set(REPARSE)
