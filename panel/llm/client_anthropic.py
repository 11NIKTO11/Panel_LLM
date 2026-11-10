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

    def __init__(self, api_key: Union[str, None] = None):
        if Anthropic is None:
            raise ImportError("anthropic package is required for Claude models.")
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY must be set for Claude models.")
        headers = {"anthropic-version": self.API_VERSION} if self.API_VERSION else None
        self.client = Anthropic(api_key=key, default_headers=headers)

    def _sampling(self, temperature: Union[float, None]) -> dict:
        """
        Raises:
            ValueError: on a set temperature, which this API version rejects.
        """
        if temperature is not None:
            raise ValueError(
                "the Anthropic API takes no temperature; declare the run with temperature=None"
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
            **self._sampling(temperature),
        )
        self.last_model = getattr(response, "model", None)
        if response.parsed_output is None:
            raise ValueError(f"no parsed output from '{model}': stop_reason={response.stop_reason}")
        return response.parsed_output

