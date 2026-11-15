import os
from typing import Type, Union

from pydantic import BaseModel

from .base import BaseLLMClient

try:
    from openai import OpenAI
    from openai.lib._pydantic import to_strict_json_schema
except Exception:  # pragma: no cover - optional dependency guard
    OpenAI = to_strict_json_schema = None


class OpenAIClient(BaseLLMClient):
    SHORT_NAME = "OpenAI"

    def __init__(self, api_key: Union[str, None] = None):
        if OpenAI is None:
            raise ImportError("openai package is required for OpenAI models.")
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY must be set for OpenAI models.")
        self.client = OpenAI(api_key=key)

    def schema_sent(self, model: str, response_format: Type[BaseModel]) -> dict:
        """The schema as the SDK's `responses.parse` rewrites it for strict mode: closed objects, every property required."""
        return to_strict_json_schema(response_format)

    def call_llm(self, model: str, prompt: str, response_format: Type[BaseModel], temperature: Union[float, None],
                 instructions: Union[str, None] = None) -> BaseModel:
        response = self.client.responses.parse(
            model=model,
            input=prompt,
            text_format=response_format,
            **({} if temperature is None else {"temperature": temperature}),
            **({"instructions": instructions} if instructions else {}),
        )
        self.last_model = getattr(response, "model", None)
        return response.output_parsed
