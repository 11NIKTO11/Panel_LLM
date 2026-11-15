"""
The clients, offline: what a call sends and what it refuses, with the SDK's HTTP transport
replaced by a stand-in - no request leaves the machine.

The Anthropic clients are checked on the request as it leaves the SDK: every field of the
body, and that the schema in it says everything the framing says.
"""
import json
from dataclasses import asdict

import httpx2
import pytest
from anthropic import Anthropic, DefaultHttpxClient

from panel.llm.client_anthropic import AnthropicClient, AnthropicLegacyClient
from panel.llm.client_openrouter import OpenRouterClient
from panel.llm.schema_compat import bounds_as_text, move_ref_descriptions
from panel.topic import framing_names, load_framing, load_topic, make_response_models, topic_names

from tests.conftest import TOPICS_DIR

PROMPT, SYSTEM = "Jsem žena, je mi 40 let. Koho jsem volila?", "Odpovídej jako respondent."
FRAMINGS = [(topic, framing) for topic in topic_names(TOPICS_DIR) for framing in framing_names(topic, TOPICS_DIR)]
CLIENTS = [(AnthropicClient, None), (AnthropicLegacyClient, 0.3)]


def framed(topic_name: str = "election_2021", framing_name: str = "cs"):
    topic = load_topic(topic_name, TOPICS_DIR)
    framing = load_framing(topic, framing_name)
    return topic, framing, make_response_models(topic, framing)


@pytest.fixture(scope="module")
def probabilities():
    topic, _, models = framed()
    return models[0], topic


def answer(model) -> dict:
    """One valid answer of a response model."""
    label = next(iter(model.LABELS))
    found = {"voted_or_not": {"voted": 0.7, "not_voted": 0.3}} if model.TURNOUT else {}
    return {**found, **({"options": [{"name": label, "probability": 1.0}]} if "options" in model.model_fields else {"choice": label})}


def answering(client, model, server_tool_use=None):
    """The client with its SDK talking to a stand-in that records each request and answers one valid response."""
    sent = []

    def reply(request):
        sent.append({"headers": dict(request.headers), "body": json.loads(request.content), "url": str(request.url)})
        usage = {"input_tokens": 1, "output_tokens": 1, **({"server_tool_use": server_tool_use} if server_tool_use else {})}
        return httpx2.Response(200, json={
            "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-test",
            "content": [{"type": "text", "text": json.dumps(answer(model))}], "stop_reason": "end_turn", "stop_sequence": None, "usage": usage,
        })

    headers = {"anthropic-version": client.API_VERSION} if client.API_VERSION else None
    client.client = Anthropic(api_key="test", default_headers=headers, http_client=DefaultHttpxClient(transport=httpx2.MockTransport(reply)))
    return sent


def nodes(schema) -> list:
    """Every object of a JSON schema, the schema itself first."""
    if isinstance(schema, dict):
        return [schema] + [node for value in schema.values() for node in nodes(value)]
    return [node for value in schema for node in nodes(value)] if isinstance(schema, list) else []


def descriptions(schema) -> str:
    return "\n".join(node["description"] for node in nodes(schema) if isinstance(node.get("description"), str))


def texts(framing, shape: str) -> list:
    """Every text of a framing's schema block that one response shape shows."""
    blocks = asdict(framing.schema)
    shown = [blocks["turnout"]] if blocks["turnout"] else []
    shown += [blocks["option"], blocks["probabilities"]] if shape == "Prob" else [blocks["pick"]]
    return [text for block in shown for key, text in block.items() if key != "title"]


@pytest.mark.parametrize("kind, temperature", CLIENTS)
@pytest.mark.parametrize("shape", [0, 1], ids=["Prob", "Pick"])
def test_the_request_is_exactly_what_the_run_declares(kind, temperature, shape):
    model = framed()[2][shape]
    client = kind(api_key="test", effort="low")
    sent = answering(client, model)
    parsed = client.call_llm("claude-test", PROMPT, model, temperature, instructions=SYSTEM)
    assert len(sent) == 1 and sent[0]["url"].endswith("/v1/messages")
    body = sent[0]["body"]
    assert body == {
        "model": "claude-test", "max_tokens": 16000, "system": SYSTEM, "tools": [],
        "messages": [{"role": "user", "content": PROMPT}],
        "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": client.schema_sent("claude-test", model)}},
        **({"temperature": temperature} if kind is AnthropicLegacyClient else {}),
    }
    assert (sent[0]["headers"]["anthropic-version"] == "2023-06-01") == (kind is AnthropicLegacyClient)
    assert parsed == model.model_validate(answer(model)) and client.last_model == "claude-test"


@pytest.mark.parametrize("kind", [AnthropicClient, AnthropicLegacyClient])
def test_what_a_run_does_not_set_is_not_sent(kind, probabilities):
    model, _ = probabilities
    client = kind(api_key="test")
    sent = answering(client, model)
    client.call_llm("claude-test", PROMPT, model, None)
    body = sent[0]["body"]
    assert set(body) == {"model", "max_tokens", "messages", "tools", "output_config"}
    assert set(body["output_config"]) == {"format"}


@pytest.mark.parametrize("kind, temperature", CLIENTS)
@pytest.mark.parametrize(("topic_name", "framing_name"), FRAMINGS, ids=[f"{topic}/{framing}" for topic, framing in FRAMINGS])
def test_the_schema_the_model_receives_says_everything_the_framing_says(kind, temperature, topic_name, framing_name):
    topic, framing, models = framed(topic_name, framing_name)
    labels = [framing.labels[option] for option in topic.options]
    for model in models:
        client = kind(api_key="test")
        sent = answering(client, model)
        client.call_llm("claude-test", PROMPT, model, temperature)
        schema = sent[0]["body"]["output_config"]["format"]["schema"]
        said = descriptions(schema)
        assert [text for text in texts(framing, model.SHORT_NAME) if text not in said] == []
        assert [node["enum"] for node in nodes(schema) if "enum" in node] == [labels]
        numbers = [node for node in nodes(schema) if node.get("type") == "number"]
        assert len(numbers) == (2 if topic.turnout else 0) + (model.SHORT_NAME == "Prob")
        for node in numbers:
            assert "minimum: 0" in node["description"] and "maximum: 1" in node["description"]
            assert "minimum" not in node and "maximum" not in node
        objects = [node for node in nodes(schema) if node.get("type") == "object"]
        assert all(node["additionalProperties"] is False and set(node["required"]) == set(node["properties"]) for node in objects)
        assert not [node for node in nodes(schema) if "$ref" in node and len(node) > 1]
        assert set(schema["properties"]) == set(model.model_fields)


def test_a_temperature_is_refused_where_the_api_takes_none(probabilities):
    model, _ = probabilities
    client = AnthropicClient(api_key="test")
    sent = answering(client, model)
    with pytest.raises(ValueError, match="temperature"):
        client.call_llm("claude-test", PROMPT, model, 0.0)
    assert not sent


def test_a_call_that_used_a_server_tool_fails(probabilities):
    model, _ = probabilities
    client = AnthropicClient(api_key="test")
    answering(client, model, server_tool_use={"web_search_requests": 1})
    with pytest.raises(ValueError, match="server tools"):
        client.call_llm("claude-test", "prompt", model, None)
    answering(client, model, server_tool_use={"web_search_requests": 0})
    assert client.call_llm("claude-test", "prompt", model, None).voted_or_not.voted == 0.7


@pytest.mark.parametrize(("topic_name", "framing_name"), FRAMINGS, ids=[f"{topic}/{framing}" for topic, framing in FRAMINGS])
@pytest.mark.parametrize("routed", ["gpt-4o", "claude-sonnet-4-5", "google/gemini-3-flash-preview"])
def test_openrouter_fixups_leave_no_text_out(routed, topic_name, framing_name):
    _, framing, models = framed(topic_name, framing_name)
    client = OpenRouterClient(api_key="test")
    for model in models:
        said = descriptions(client.schema_sent(routed, model))
        assert [text for text in texts(framing, model.SHORT_NAME) if text not in said] == []


def test_a_description_beside_a_ref_moves_to_its_definition():
    schema = {
        "properties": {"a": {"$ref": "#/$defs/A", "description": "of the field"}, "b": {"$ref": "#/$defs/B", "description": "alone"}},
        "$defs": {"A": {"type": "object", "description": "of the definition"}, "B": {"type": "object"}},
    }
    moved = move_ref_descriptions(schema)
    assert moved["properties"] == {"a": {"$ref": "#/$defs/A"}, "b": {"$ref": "#/$defs/B"}}
    assert moved["$defs"]["A"]["description"] == "of the definition\n\nof the field" and moved["$defs"]["B"]["description"] == "alone"
    with pytest.raises(ValueError, match="no such definition"):
        move_ref_descriptions({"properties": {"a": {"$ref": "#/$defs/A", "description": "lost"}}})


def test_bounds_become_text():
    schema = {"properties": {"p": {"type": "number", "minimum": 0, "maximum": 1, "description": "a share"}, "n": {"type": "integer", "minimum": 1}, "s": {"type": "string"}}}
    assert bounds_as_text(schema)["properties"] == {
        "p": {"type": "number", "description": "a share\n\n{minimum: 0, maximum: 1}"}, "n": {"type": "integer", "description": "{minimum: 1}"}, "s": {"type": "string"},
    }
