"""
The clients, offline: what a call sends and what it refuses, with the SDK's HTTP transport
replaced by a stand-in - no request leaves the machine.

The Anthropic clients are checked on the request as it leaves the SDK: every field of the
body, and that the schema in it says everything the framing says. OpenRouter's requests are
checked where they are handed to `requests`.
"""
import json
import time
from dataclasses import asdict
from types import SimpleNamespace

import httpx2
import pytest
from anthropic import Anthropic, DefaultHttpxClient

import pandas as pd
import requests

from panel.llm import Batch, BatchFailed, BatchStopped, CallFailed, Flaw, RateLimited, client_openrouter, ordered_efforts, run_llm_batch
from panel.llm.client_gemini import GeminiClient
from panel.llm.client_openai import OpenAIClient
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
def test_the_client_reports_the_parameters_it_sends(kind, probabilities):
    model, _ = probabilities
    client = kind(api_key="test", effort="low", max_tokens=2000)
    sent = answering(client, model)
    client.call_llm("claude-test", PROMPT, model, None)
    body = sent[0]["body"]
    assert client.parameters() == {"max_tokens": 2000, "effort": "low", "api_version": kind.API_VERSION, "tools": []}
    assert (body["max_tokens"], body["output_config"]["effort"], body["tools"]) == (2000, "low", [])


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
    with pytest.raises(CallFailed, match="server tools"):
        client.call_llm("claude-test", "prompt", model, None)
    answering(client, model, server_tool_use={"web_search_requests": 0})
    assert client.call_llm("claude-test", "prompt", model, None).voted_or_not.voted == 0.7


@pytest.mark.parametrize("kind", [AnthropicClient, AnthropicLegacyClient])
@pytest.mark.parametrize("effort, levels", [
    ({"supported": True, "max": {"supported": True}, "low": {"supported": True}, "medium": {"supported": True},
      "high": {"supported": True}, "xhigh": {"supported": False}}, ("low", "medium", "high", "max")),
    ({"supported": False, "low": {"supported": False}, "medium": {"supported": False}, "high": {"supported": False},
      "max": {"supported": False}, "xhigh": None}, ()),
], ids=["some", "none"])
def test_anthropic_reports_the_efforts_a_model_takes(kind, effort, levels):
    asked = []

    def reply(request):
        asked.append((request.method, request.url.path))
        return httpx2.Response(200, json={"id": "claude-test", "type": "model", "display_name": "Test", "created_at": "2026-01-01T00:00:00Z", "capabilities": {"effort": effort}})

    client = kind(api_key="test")
    client.client = Anthropic(api_key="test", http_client=DefaultHttpxClient(transport=httpx2.MockTransport(reply)))
    assert client.efforts("claude-test") == levels and asked == [("GET", "/v1/models/claude-test")]


class Reply:
    """What `requests` hands back, as far as the OpenRouter client reads it."""

    def __init__(self, payload: dict, status_code: int = 200):
        self.payload, self.status_code, self.text = payload, status_code, json.dumps(payload)

    def json(self) -> dict:
        return self.payload


CATALOGUE = {"data": [
    {"id": "openai/gpt-5", "reasoning": {"mandatory": True, "supported_efforts": ["high", "medium", "low", "minimal"]}},
    {"id": "openai/gpt-4o"},
    {"id": "anthropic/claude-sonnet-4.5", "reasoning": {"mandatory": False}},
    {"id": "google/any-level", "reasoning": {"supported_efforts": None}},
    {"id": "google/new-level", "reasoning": {"supported_efforts": ["high", "ultra"]}},
]}


def test_openrouter_reports_the_efforts_a_model_takes(monkeypatch):
    monkeypatch.setattr(client_openrouter.requests, "get", lambda url, **kwargs: Reply(CATALOGUE))
    client = OpenRouterClient(api_key="test")
    assert client.efforts("gpt-5") == ("minimal", "low", "medium", "high")
    assert client.efforts("gpt-4o") == () and client.efforts("claude-sonnet-4-5") == ()
    assert client.efforts("google/any-level") == ("none", "minimal", "low", "medium", "high", "xhigh", "max")
    with pytest.raises(ValueError, match="ultra"):
        client.efforts("google/new-level")
    with pytest.raises(ValueError, match="no 'google/nobody'"):
        client.efforts("google/nobody")
    assert ordered_efforts(["max", "low"]) == ("low", "max")


@pytest.mark.parametrize("settings, sent, reported", [
    ({}, {}, {"effort": None, "max_tokens": None}),
    ({"effort": "low"}, {"reasoning": {"effort": "low"}, "max_tokens": 16000}, {"effort": "low", "max_tokens": 16000}),
    ({"effort": "high", "max_tokens": 4000}, {"reasoning": {"effort": "high"}, "max_tokens": 4000}, {"effort": "high", "max_tokens": 4000}),
    ({"max_tokens": 4000}, {"max_tokens": 4000}, {"effort": None, "max_tokens": 4000}),
], ids=["plain", "effort", "effort and limit", "limit"])
def test_openrouter_sends_the_effort_it_was_built_with(monkeypatch, probabilities, settings, sent, reported):
    model, _ = probabilities
    bodies = []

    def post(url, **request):
        bodies.append(request["json"])
        return Reply({"model": "openai/gpt-5", "choices": [{"message": {"content": json.dumps(answer(model))}}]})

    monkeypatch.setattr(client_openrouter.requests, "post", post)
    client = OpenRouterClient(api_key="test", **settings)
    client.call_llm("gpt-5", PROMPT, model, None, instructions=SYSTEM)
    body = bodies[0]
    assert {key: body[key] for key in body if key in ("reasoning", "max_tokens", "temperature")} == sent
    assert client.parameters() == {"timeout": 120.0, "strict": True, **reported}


@pytest.mark.parametrize(("topic_name", "framing_name"), FRAMINGS, ids=[f"{topic}/{framing}" for topic, framing in FRAMINGS])
@pytest.mark.parametrize("routed", ["gpt-4o", "claude-sonnet-4-5", "google/gemini-3-flash-preview"])
def test_openrouter_fixups_leave_no_text_out(routed, topic_name, framing_name):
    _, framing, models = framed(topic_name, framing_name)
    client = OpenRouterClient(api_key="test")
    for model in models:
        said = descriptions(client.schema_sent(routed, model))
        assert [text for text in texts(framing, model.SHORT_NAME) if text not in said] == []


FAILURES = [(429, RateLimited), (500, CallFailed), (529, CallFailed), (408, CallFailed), (401, BatchFailed), (400, BatchFailed), (404, BatchFailed)]


@pytest.mark.parametrize("kind", [AnthropicClient, AnthropicLegacyClient])
@pytest.mark.parametrize("status, failure", FAILURES)
def test_anthropic_names_each_failure(kind, status, failure, probabilities):
    model, _ = probabilities
    client = kind(api_key="test")
    reply = lambda request: httpx2.Response(status, json={"type": "error", "error": {"type": "error", "message": "no"}})
    client.client = Anthropic(api_key="test", max_retries=0, http_client=DefaultHttpxClient(transport=httpx2.MockTransport(reply)))
    with pytest.raises(failure):
        client.call_llm("claude-test", PROMPT, model, None)


def test_anthropic_names_a_lost_connection_and_an_unusable_answer(probabilities):
    model, _ = probabilities
    client = AnthropicClient(api_key="test")

    def lost(request):
        raise httpx2.ConnectError("no route")

    client.client = Anthropic(api_key="test", max_retries=0, http_client=DefaultHttpxClient(transport=httpx2.MockTransport(lost)))
    with pytest.raises(CallFailed):
        client.call_llm("claude-test", PROMPT, model, None)
    wrong = lambda request: httpx2.Response(200, json={
        "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-test", "content": [{"type": "text", "text": "{}"}],
        "stop_reason": "end_turn", "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1},
    })
    client.client = Anthropic(api_key="test", max_retries=0, http_client=DefaultHttpxClient(transport=httpx2.MockTransport(wrong)))
    with pytest.raises(CallFailed):
        client.call_llm("claude-test", PROMPT, model, None)


@pytest.mark.parametrize("status, failure", FAILURES)
def test_openrouter_names_each_failure(monkeypatch, status, failure, probabilities):
    model, _ = probabilities
    client = OpenRouterClient(api_key="test")
    monkeypatch.setattr(client_openrouter.requests, "post", lambda url, **request: Reply({"error": "no"}, status))
    with pytest.raises(failure):
        client.call_llm("gpt-4o", PROMPT, model, 0.0)
    monkeypatch.setattr(client_openrouter.requests, "post", lambda url, **request: Reply({"error": {"code": status, "message": "no"}}))
    with pytest.raises(failure):
        client.call_llm("gpt-4o", PROMPT, model, 0.0)


@pytest.mark.parametrize("payload", [
    {"error": "no code"}, {"choices": []}, {"choices": [{"message": {"content": "not json"}}]}, {"choices": [{"message": {"content": "{}"}}]},
    {"choices": [{"message": {"content": None}}]},
], ids=["error without a code", "no choice", "not json", "not the schema", "no content"])
def test_openrouter_names_an_unusable_answer(monkeypatch, payload, probabilities):
    model, _ = probabilities
    client = OpenRouterClient(api_key="test")
    monkeypatch.setattr(client_openrouter.requests, "post", lambda url, **request: Reply(payload))
    with pytest.raises(CallFailed):
        client.call_llm("gpt-4o", PROMPT, model, 0.0)

    def timed_out(url, **request):
        raise requests.Timeout("too slow")

    monkeypatch.setattr(client_openrouter.requests, "post", timed_out)
    with pytest.raises(CallFailed, match="Timeout"):
        client.call_llm("gpt-4o", PROMPT, model, 0.0)


def test_openai_and_gemini_never_answer_with_nothing(probabilities):
    model, _ = probabilities
    client = OpenAIClient(api_key="test")
    client.client = SimpleNamespace(responses=SimpleNamespace(parse=lambda **request: SimpleNamespace(model="gpt-test", output_parsed=None)))
    with pytest.raises(CallFailed, match="no parsed output"):
        client.call_llm("gpt-test", PROMPT, model, 0.0)
    gemini = GeminiClient(api_key="test")
    gemini.client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **request: SimpleNamespace(model_version="gemini-test", text=None)))
    with pytest.raises(CallFailed, match="no text"):
        gemini.call_llm("gemini-test", PROMPT, model, 0.0)
    gemini.client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **request: SimpleNamespace(model_version="gemini-test", text="{}")))
    with pytest.raises(CallFailed, match="not the schema"):
        gemini.call_llm("gemini-test", PROMPT, model, 0.0)


class Failing:
    """A client whose calls raise what it was given, in turn, and then answer."""

    def __init__(self, *failures, takes: float = 0.0):
        self.failures, self.calls, self.takes = list(failures), 0, takes

    def call_llm(self, model, prompt, response_format, temperature, instructions=None):
        self.calls += 1
        time.sleep(self.takes)
        if self.failures:
            failure = self.failures.pop(0)
            if failure is not None:
                raise failure
            return None
        return "answer"


def batch(client, rows: int = 1, retries: int = 3, review=None):
    data = pd.DataFrame({"text": [f"prompt {row}" for row in range(rows)]})
    return run_llm_batch(data, prompt_creator=lambda row: row["text"], response_model=None, model="m", temperature=None,
                         client=client, instructions=None, review=review, max_workers=1, retries=retries)


def test_the_runner_waits_on_a_rate_limit_and_tries_a_failed_call_again(monkeypatch):
    waits = []
    monkeypatch.setattr("panel.llm.runner.time.sleep", waits.append)
    client = Failing(RateLimited("slow down"), CallFailed("lost"))
    assert batch(client) == Batch({0: ("prompt 0", "answer")}, {}, {0: ("rate_limited", "failed")}, []) and client.calls == 3
    assert len(waits := [wait for wait in waits if wait]) == 1 and 1 <= waits[0] < 2
    assert batch(Failing(CallFailed("a"), CallFailed("b")), retries=2) == Batch({}, {}, {0: ("failed", "failed")}, [0])
    assert batch(Failing()) == Batch({0: ("prompt 0", "answer")})


@pytest.mark.parametrize("failure", [BatchFailed("wrong key"), KeyError("a bug"), None], ids=["batch failed", "unforeseen", "answered None"])
def test_the_runner_stops_the_batch_on_what_no_retry_mends(failure):
    client = Failing(*[failure] * 50, takes=0.01)
    with pytest.raises(BatchStopped) as stopped:
        batch(client, rows=50)
    assert client.calls < 50 and isinstance(stopped.value.__cause__, type(failure) if failure else TypeError)


def test_a_stopped_batch_hands_back_what_came_before():
    client, prompts = Failing(), []

    def call(model, prompt, response_format, temperature, instructions=None):
        prompts.append(prompt)
        time.sleep(0.01)
        if len(prompts) == 4:
            raise BatchFailed("no credit")
        return f"answer to {prompt}"

    client.call_llm = call
    with pytest.raises(BatchStopped) as stopped:
        batch(client, rows=20)
    results = stopped.value.batch.results
    assert len(prompts) < 20 and 3 not in results and {0, 1, 2} <= set(results)
    assert results == {row: (f"prompt {row}", f"answer to prompt {row}") for row in results}


class Answering:
    """A client that answers each call with the next of the answers it was given."""

    def __init__(self, *answers):
        self.answers = list(answers)

    def call_llm(self, model, prompt, response_format, temperature, instructions=None):
        return self.answers.pop(0)


def test_the_runner_keeps_the_best_flawed_answer_and_no_unusable_one():
    def review(answer):
        return None if answer == "good" else Flaw(reasons=(answer,), usable=answer != "broken", distance={"far": 0.5, "near": 0.1}.get(answer, 9.0))

    assert batch(Answering("far", "good"), review=review) == Batch({0: ("prompt 0", "good")}, {}, {0: ("far",)}, [])
    assert batch(Answering("far", "near", "broken"), review=review) == Batch({0: ("prompt 0", "near")}, {0: ("near",)}, {0: ("far", "near", "broken")}, [])
    assert batch(Answering("near", "far", "far"), review=review) == Batch({0: ("prompt 0", "near")}, {0: ("near",)}, {0: ("near", "far", "far")}, [])
    assert batch(Answering("broken", "broken", "broken"), review=review) == Batch({}, {}, {0: ("broken",) * 3}, [0])


def test_the_runner_states_no_default():
    with pytest.raises(TypeError):
        run_llm_batch(pd.DataFrame({"text": ["a"]}), prompt_creator=lambda row: row["text"], response_model=None, client=Failing(), review=None)


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
