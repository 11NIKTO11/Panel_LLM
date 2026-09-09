"""Response schemas are prompt content: each topic and framing generates exactly the schema its runs are shown."""
import json
import shutil

import pandas as pd
import pytest

from panel.topic import framing_names, load_framing, load_topic, make_response_models, topic_names

from tests.conftest import LEGACY_TOPIC, RESULTS_DIR, TOPICS_DIR, load_fixture, require
from tests.snapshots import schema_snapshot

PAIRS = [(topic, framing) for topic in topic_names(TOPICS_DIR) for framing in framing_names(topic, TOPICS_DIR)]


def generated(topic_name: str, framing_name: str) -> dict:
    topic = load_topic(topic_name, TOPICS_DIR)
    return schema_snapshot(*make_response_models(topic, load_framing(topic, framing_name, TOPICS_DIR)))


@pytest.mark.parametrize(("topic_name", "framing_name"), PAIRS, ids=[f"{topic}/{framing}" for topic, framing in PAIRS])
def test_schema_matches_snapshot(topic_name, framing_name):
    assert generated(topic_name, framing_name) == load_fixture(f"schemas/{topic_name}/{framing_name}")


def test_legacy_schema_identical():
    config = require(RESULTS_DIR / LEGACY_TOPIC / "config.json")
    frozen = json.loads(config.read_text(encoding="utf-8"))["schemas"]
    assert frozen == load_fixture(f"schemas/{LEGACY_TOPIC}/cs")
    assert json.dumps(generated(LEGACY_TOPIC, "cs"), ensure_ascii=False) == json.dumps(frozen, ensure_ascii=False)


def test_topic_loads_all():
    topics = [load_topic(name, TOPICS_DIR) for name in topic_names(TOPICS_DIR)]
    assert any(topic.turnout is None for topic in topics), "no topic exercises the turnout-less shape"
    for topic in topics:
        assert all(option.isascii() for option in topic.options)
        assert topic.columns == list(topic.turnout or ()) + list(topic.options)
        framings = [load_framing(topic, name, TOPICS_DIR) for name in framing_names(topic.name, TOPICS_DIR)]
        assert framings, topic.name
        for framing in framings:
            assert set(framing.labels) == set(topic.options)
            for model in make_response_models(topic, framing):
                assert ("voted_or_not" in model.model_fields) == (topic.turnout is not None), topic.name


def test_rename_maps_names_and_aliases():
    topic = load_topic("election_2021", TOPICS_DIR)
    frame = pd.DataFrame(columns=[topic.names[option] for option in topic.options] + ["p_voted", "Prompt"])
    assert list(topic.rename(frame).columns) == list(topic.options) + ["Voted", "Prompt"]


def test_to_series_is_keyed_by_id():
    topic = load_topic("election_2021", TOPICS_DIR)
    framing = load_framing(topic, "cs", TOPICS_DIR)
    probabilities, pick = make_response_models(topic, framing)
    first = topic.options[0]
    row = probabilities(voted_or_not={"voted": 0.7, "not_voted": 0.3}, parties=[{"name": topic.names[first], "probability": 1.0}]).to_series()
    assert list(row.index) == topic.columns and row[first] == 1.0 and row[topic.options[1]] == 0.0
    chosen = pick(voted_or_not={"voted": 0.7, "not_voted": 0.3}, choice=topic.names[first]).to_series()
    assert chosen[first] == 1 and chosen[topic.options[-1]] == 0


@pytest.mark.parametrize("bad_labels", [{"Nobody": "x"}, {"ANO": "Jiná strana"}], ids=["unknown id", "duplicate label"])
def test_framing_rejects_bad_labels(tmp_path, bad_labels):
    shutil.copytree(TOPICS_DIR / "election_2021", tmp_path / "election_2021")
    path = tmp_path / "election_2021" / "cs.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["labels"] = bad_labels
    path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError):
        load_framing(load_topic("election_2021", tmp_path), "cs", tmp_path)
