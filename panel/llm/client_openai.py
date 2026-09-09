import os
from typing import Type, Union

from pydantic import BaseModel

from .base import BaseLLMClient

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency guard
    OpenAI = None


class OpenAIClient(BaseLLMClient):
    SHORT_NAME = "OpenAI"

    def __init__(self, api_key: Union[str, None] = None):
        if OpenAI is None:
            raise ImportError("openai package is required for OpenAI models.")
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY must be set for OpenAI models.")
        self.client = OpenAI(api_key=key)

    def call_llm(self, model: str, prompt: str, response_format: Type[BaseModel], temperature: float,
                 instructions: Union[str, None] = None) -> BaseModel:
        response = self.client.responses.parse(
            model=model,
            temperature=temperature,
            input=prompt,
            text_format=response_format,
            **({"instructions": instructions} if instructions else {}),
        )
        self.last_model = getattr(response, "model", None)
        return response.output_parsed
