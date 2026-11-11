import os
from typing import Type, Union

from pydantic import BaseModel

from .base import BaseLLMClient

try:
    from anthropic import Anthropic
except Exception:  # pragma: no cover - optional dependency guard
    Anthropic = None


class AnthropicClient(BaseLLMClient):
    """Claude on the API version the installed SDK sends, which takes no temperature."""
    SHORT_NAME = "Anthropic"
    API_VERSION: Union[str, None] = None

    def __init__(self, api_key: Union[str, None] = None, effort: Union[str, None] = None):
        """
        Args:
            api_key: Anthropic key; falls back to ANTHROPIC_API_KEY.
            effort: `low` .. `max`, how much the model thinks per call; None leaves the model's default.
        """
        if Anthropic is None:
            raise ImportError("anthropic package is required for Claude models.")
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY must be set for Claude models.")
        headers = {"anthropic-version": self.API_VERSION} if self.API_VERSION else None
        self.client = Anthropic(api_key=key, default_headers=headers)
        self.effort = effort

    def _sampling(self, temperature: Union[float, None]) -> dict:
        """
        Raises:
            ValueError: on a set temperature, which this API version rejects.
        """
        if temperature is not None:
            raise ValueError(
                "the Anthropic API takes no temperature; declare the run with temperature=None, "
                "or with client='AnthropicLegacy' on a model that still accepts one"
            )
        return {}

    def call_llm(self, model: str, prompt: str, response_format: Type[BaseModel], temperature: Union[float, None],
                 instructions: Union[str, None] = None) -> BaseModel:
        response = self.client.messages.parse(
            model=model,
            max_tokens=16000,
            messages=[{"role": "user", "content": prompt}],
            output_format=response_format,
            **({"system": instructions} if instructions else {}),
            **({"output_config": {"effort": self.effort}} if self.effort else {}),
            **self._sampling(temperature),
        )
        self.last_model = getattr(response, "model", None)
        if response.parsed_output is None:
            raise ValueError(f"no parsed output from '{model}': stop_reason={response.stop_reason}")
        return response.parsed_output


class AnthropicLegacyClient(AnthropicClient):
    """
    Claude on API version 2023-06-01, the last that accepts a temperature.

    For the models that still honour one there (Opus 4.6, Sonnet 4.6, Haiku 4.5, Opus 4.5);
    Opus 4.7 and later refuse it on any version. The installed SDK's own call has no
    temperature argument any more, so it travels in the request body.
    """
    SHORT_NAME = "AnthropicLegacy"
    API_VERSION = "2023-06-01"

    def _sampling(self, temperature: Union[float, None]) -> dict:
        return {} if temperature is None else {"extra_body": {"temperature": temperature}}
