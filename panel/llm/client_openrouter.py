import json
import os
from typing import Tuple, Type, Union

import requests
from pydantic import BaseModel, ValidationError

from .base import ANTHROPIC, EFFORT_ORDER, OPENAI, BaseLLMClient, detect_provider, ordered_efforts
from .errors import CallFailed, status_error
from .schema_compat import bounds_as_text, move_ref_descriptions, set_additional_properties_false


class OpenRouterClient(BaseLLMClient):
    """
    Any provider's model through OpenRouter, which takes one `reasoning.effort` for all of them and
    turns it into the routed provider's own setting: the same word for OpenAI, the nearest
    `thinkingLevel` for Gemini, a thinking budget that is a share of `max_tokens` for Claude.
    It moves a level a model lacks to the nearest one without saying so; `efforts` is how a
    caller refuses instead.
    """
    SHORT_NAME = "OpenRouter"
    TAKES_EFFORT = True
    EFFORT_MAX_TOKENS = 16000

    def __init__(self, api_key: Union[str, None] = None, timeout: float = 120.0, effort: Union[str, None] = None,
                 max_tokens: Union[int, None] = None):
        """
        Without a `timeout` (seconds) a hung request holds a worker thread of the batch runner's pool
        forever and the batch never finishes; a timeout is a `CallFailed` instead, and is tried again.
        `max_tokens` is the most a call may write, thinking included; None sends none, except with an
        `effort`, where `EFFORT_MAX_TOKENS` is sent so that a budget which is a share of it has a known base.
        """
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY must be set for OpenRouter.")
        self.header = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.models_url = "https://openrouter.ai/api/v1/models"
        self.timeout = timeout
        self.effort = effort
        self.max_tokens = self.EFFORT_MAX_TOKENS if effort and max_tokens is None else max_tokens

    def parameters(self) -> dict:
        return {"timeout": self.timeout, "strict": True, "effort": self.effort, "max_tokens": self.max_tokens}

    def efforts(self, model: str) -> Tuple[str, ...]:
        """
        From the model catalogue's `reasoning.supported_efforts`: no such key is a model without an
        effort setting, null one that takes every level OpenRouter names.
        """
        routed = "/".join(self._route(model))
        response = requests.get(self.models_url, headers=self.header, timeout=self.timeout)
        if response.status_code != 200:
            raise ValueError(f"OpenRouter's model catalogue failed with HTTP {response.status_code}: {response.text[:500]}")
        found = [entry for entry in response.json()["data"] if entry["id"] == routed]
        if not found:
            raise ValueError(f"OpenRouter's model catalogue has no '{routed}'")
        reasoning = found[0].get("reasoning") or {}
        if "supported_efforts" not in reasoning:
            return ()
        return EFFORT_ORDER if reasoning["supported_efforts"] is None else ordered_efforts(reasoning["supported_efforts"])

    def call_llm(self, model: str, prompt: str, response_format: Type[BaseModel], temperature: Union[float, None],
                 instructions: Union[str, None] = None) -> BaseModel:
        provider, model_id = self._route(model)
        response_format_json = self.schema_sent(model, response_format)

        try:
            response = self._post(provider, model_id, prompt, response_format, response_format_json, temperature, instructions)
        except requests.RequestException as error:
            raise CallFailed(f"{type(error).__name__}: {error}") from error
        if response.status_code != 200:
            raise status_error(
                response.status_code,
                f"OpenRouter request for '{provider}/{model_id}' failed with HTTP {response.status_code}: {response.text[:500]}",
            )
        try:
            data = response.json()
        except ValueError as error:
            raise CallFailed(f"OpenRouter answered '{provider}/{model_id}' with no JSON: {response.text[:200]}") from error
        self.last_model = data.get("model")
        if "error" in data:  # OpenRouter can return HTTP 200 with an error payload
            code = data["error"].get("code") if isinstance(data["error"], dict) else None
            message = f"OpenRouter returned an error for '{provider}/{model_id}': {data['error']}"
            raise status_error(code, message) if isinstance(code, int) else CallFailed(message)
        try:
            return response_format(**json.loads(data["choices"][0]["message"]["content"]))
        except (KeyError, IndexError, TypeError, ValueError, ValidationError) as error:
            raise CallFailed(f"answer of '{provider}/{model_id}' is not the schema: {str(data)[:500]}") from error

    def _post(self, provider: str, model_id: str, prompt: str, response_format: Type[BaseModel], schema: dict,
              temperature: Union[float, None], instructions: Union[str, None]) -> requests.Response:
        return requests.post(
            self.url,
            headers=self.header,
            json={
                "model": f"{provider}/{model_id}",
                "messages": (
                    [{"role": "system", "content": instructions}] if instructions else []
                ) + [{"role": "user", "content": prompt}],
                **({} if temperature is None else {"temperature": temperature}),
                **({"reasoning": {"effort": self.effort}} if self.effort else {}),
                **({} if self.max_tokens is None else {"max_tokens": self.max_tokens}),
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_format.__name__,
                        "strict": True,
                        "schema": schema,
                    }
                },
            },
            timeout=self.timeout,
        )

    def schema_sent(self, model: str, response_format: Type[BaseModel]) -> dict:
        """The model's schema with the fix-ups the routed provider's strict mode needs; nothing it says is left out."""
        provider, _ = self._route(model)
        schema = response_format.model_json_schema()
        if provider == OPENAI:
            return move_ref_descriptions(set_additional_properties_false(schema))
        if provider == ANTHROPIC:
            return bounds_as_text(set_additional_properties_false(schema))
        return schema

    def _route(self, model: str) -> Tuple[str, str]:
        if "/" in model:
            provider, model_id = model.split("/", 1)
            return provider, model_id
        provider = detect_provider(model)
        if not provider:
            raise ValueError(
                f"Cannot infer OpenRouter provider for model '{model}'. "
                "Pass a full id like 'openai/gpt-4o-2024-11-20'."
            )
        return provider, self._fix_model(model)

    def _fix_model(self, model: str) -> str:
        # OpenRouter ids write Anthropic versions with a dot: claude-sonnet-4.5
        if model == "claude-sonnet-4-5":
            model = "claude-sonnet-4.5"
        return model
