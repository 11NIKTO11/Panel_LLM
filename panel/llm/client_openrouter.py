import json
import os
from typing import Tuple, Type, Union

import requests
from pydantic import BaseModel

from .base import ANTHROPIC, OPENAI, BaseLLMClient, detect_provider
from .schema_compat import bounds_as_text, move_ref_descriptions, set_additional_properties_false


class OpenRouterClient(BaseLLMClient):
    SHORT_NAME = "OpenRouter"

    def __init__(self, api_key: Union[str, None] = None, timeout: float = 120.0):
        """
        Without a `timeout` (seconds) a hung request holds a worker thread of the batch runner's pool
        forever and the batch never finishes; a timeout surfaces as a retryable error instead.
        """
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY must be set for OpenRouter.")
        self.header = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.timeout = timeout

    def parameters(self) -> dict:
        return {"timeout": self.timeout, "strict": True}

    def call_llm(self, model: str, prompt: str, response_format: Type[BaseModel], temperature: Union[float, None],
                 instructions: Union[str, None] = None) -> BaseModel:
        provider, model_id = self._route(model)
        response_format_json = self.schema_sent(model, response_format)

        response = requests.post(
            self.url,
            headers=self.header,
            json={
                "model": f"{provider}/{model_id}",
                "messages": (
                    [{"role": "system", "content": instructions}] if instructions else []
                ) + [{"role": "user", "content": prompt}],
                **({} if temperature is None else {"temperature": temperature}),
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_format.__name__,
                        "strict": True,
                        "schema": response_format_json,
                    }
                },
            },
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise ValueError(
                f"OpenRouter request for '{provider}/{model_id}' failed with HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )
        data = response.json()
        self.last_model = data.get("model")
        if "error" in data:  # OpenRouter can return HTTP 200 with an error payload
            raise ValueError(f"OpenRouter returned an error for '{provider}/{model_id}': {data['error']}")
        try:
            text = data["choices"][0]["message"]["content"]
            parsed = json.loads(text)
            return response_format(**parsed)
        except Exception:
            raise ValueError(f"Failed to parse response: {data}")

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
