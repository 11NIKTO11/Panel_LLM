import json
import os
from typing import Tuple, Type, Union

import requests
from pydantic import BaseModel

from .base import ANTHROPIC, OPENAI, BaseLLMClient, detect_provider
from .schema_compat import move_ref_descriptions, remove_min_max, set_additional_properties_false


class OpenRouterClient(BaseLLMClient):
    def __init__(self, api_key: Union[str, None] = None):
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY must be set for OpenRouter.")
        self.header = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def call_llm(self, model: str, prompt: str, response_format: Type[BaseModel], temperature: float) -> BaseModel:
        provider, model_id = self._route(model)
        response_format_json = response_format.model_json_schema()

        if provider == OPENAI:
            response_format_json = set_additional_properties_false(response_format_json)
            response_format_json = move_ref_descriptions(response_format_json)
        elif provider == ANTHROPIC:
            response_format_json = set_additional_properties_false(response_format_json)
            response_format_json = remove_min_max(response_format_json)

        response = requests.post(
            self.url,
            headers=self.header,
            json={
                "model": f"{provider}/{model_id}",
                "messages": [
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_format.__name__,
                        "strict": True,
                        "schema": response_format_json,
                    }
                },
            },
        )
        if response.status_code != 200:
            raise ValueError(
                f"OpenRouter request for '{provider}/{model_id}' failed with HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )
        data = response.json()
        if "error" in data:  # OpenRouter can return HTTP 200 with an error payload
            raise ValueError(f"OpenRouter returned an error for '{provider}/{model_id}': {data['error']}")
        try:
            text = data["choices"][0]["message"]["content"]
            parsed = json.loads(text)
            return response_format(**parsed)
        except Exception:
            raise ValueError(f"Failed to parse response: {data}")

    def _route(self, model: str) -> Tuple[str, str]:
        """Split a model name into (provider, model_id); accepts full 'provider/model' ids as-is."""
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
