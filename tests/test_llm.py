"""
The Anthropic clients, offline: what a call sends and what it refuses, with the SDK's HTTP
transport replaced by a stand-in - no request leaves the machine.
"""
import json

import httpx2
import pytest
from anthropic import Anthropic, DefaultHttpxClient

from panel.llm.client_anthropic import AnthropicClient, AnthropicLegacyClient
from panel.topic import load_framing, load_topic, make_response_models

from tests.conftest import TOPICS_DIR


@pytest.fixture(scope="module")
def probabilities():
    topic = load_topic("election_2021", TOPICS_DIR)
    return make_response_models(topic, load_framing(topic, "cs", TOPICS_DIR))[0], topic


def answering(client, probabilities, topic, server_tool_use=None):
    """The client with its SDK talking to a stand-in that records each request and answers one valid response."""
    sent = []
    answer = {"voted_or_not": {"voted": 0.7, "not_voted": 0.3}, "options": [{"name": topic.names[topic.options[0]], "probability": 1.0}]}

    def reply(request):
        sent.append({"headers": dict(request.headers), "body": json.loads(request.content)})
        usage = {"input_tokens": 1, "output_tokens": 1, **({"server_tool_use": server_tool_use} if server_tool_use else {})}
        return httpx2.Response(200, json={
            "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-test",
            "content": [{"type": "text", "text": json.dumps(answer)}], "stop_reason": "end_turn", "stop_sequence": None, "usage": usage,
        })

    headers = {"anthropic-version": client.API_VERSION} if client.API_VERSION else None
    client.client = Anthropic(api_key="test", default_headers=headers, http_client=DefaultHttpxClient(transport=httpx2.MockTransport(reply)))
    return sent


@pytest.mark.parametrize("kind, temperature", [(AnthropicClient, None), (AnthropicLegacyClient, 0.0)])
def test_a_call_offers_no_tools_and_sends_the_schema_in_output_config(kind, temperature, probabilities):
    model, topic = probabilities
    client = kind(api_key="test", effort="low")
    sent = answering(client, model, topic)
    parsed = client.call_llm("claude-test", "prompt", model, temperature, instructions="system")
    body = sent[0]["body"]
    assert body["tools"] == [] and "output_format" not in body
    assert body["output_config"]["effort"] == "low" and body["output_config"]["format"]["type"] == "json_schema"
    assert body["system"] == "system" and parsed.voted_or_not.voted == 0.7
    assert ("temperature" in body) == (kind is AnthropicLegacyClient)
    if kind is AnthropicLegacyClient:
        assert sent[0]["headers"]["anthropic-version"] == "2023-06-01"


def test_a_call_that_used_a_server_tool_fails(probabilities):
    model, topic = probabilities
    client = AnthropicClient(api_key="test")
    answering(client, model, topic, server_tool_use={"web_search_requests": 1})
    with pytest.raises(ValueError, match="server tools"):
        client.call_llm("claude-test", "prompt", model, None)
    answering(client, model, topic, server_tool_use={"web_search_requests": 0})
    assert client.call_llm("claude-test", "prompt", model, None).voted_or_not.voted == 0.7
