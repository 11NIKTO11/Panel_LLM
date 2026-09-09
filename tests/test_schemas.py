"""Response schemas are prompt content: each topic generates exactly the schema its runs are shown."""
import json

import pytest

from utils.topic import load_topic
from utils.voting import make_response_models

from tests.conftest import LEGACY_TOPIC, RESULTS_DIR, TOPICS_DIR, load_fixture, require, topic_names
from tests.snapshots import schema_snapshot

TOPICS = topic_names()


def generated(topic_name: str) -> dict:
    return schema_snapshot(*make_response_models(load_topic(topic_name, TOPICS_DIR)))


@pytest.mark.parametrize("topic_name", [name for name in TOPICS if name != LEGACY_TOPIC])
def test_schema_matches_snapshot(topic_name):
    assert generated(topic_name) == load_fixture(f"schemas/{topic_name}")


@pytest.mark.xfail(strict=True, reason="phase 4: the legacy framing has to carry the schema text those runs saw")
def test_legacy_schema_identical():
    assert generated(LEGACY_TOPIC) == load_fixture(f"schemas/{LEGACY_TOPIC}")


def test_legacy_schema_fixture_is_the_frozen_one():
    config = require(RESULTS_DIR / LEGACY_TOPIC / "config.json")
    assert json.loads(config.read_text(encoding="utf-8"))["schemas"] == load_fixture(f"schemas/{LEGACY_TOPIC}")


def test_topic_loads_all():
    topics = [load_topic(name, TOPICS_DIR) for name in TOPICS]
    assert any(topic.turnout is None for topic in topics), "no topic exercises the turnout-less shape"
    for topic in topics:
        for model in make_response_models(topic):
            assert ("voted_or_not" in model.model_fields) == (topic.turnout is not None), topic.name
