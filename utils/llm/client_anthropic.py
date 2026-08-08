import os
from typing import Type, Union

from pydantic import BaseModel

from .base import BaseLLMClient

try:
    from anthropic import Anthropic
except Exception:  # pragma: no cover - optional dependency guard
    Anthropic = None


class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key: Union[str, None] = None):
        if Anthropic is None:
            raise ImportError("anthropic package is required for Claude models.")
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY must be set for Claude models.")
        self.client = Anthropic(api_key=key)

    def call_llm(self, model: str, prompt: str, response_format: Type[BaseModel], temperature: float) -> BaseModel:
        response = self.client.beta.messages.parse(
            model=model,
            max_tokens=2048,
            temperature=temperature,
            betas=["structured-outputs-2025-11-13"],
            messages=[{"role": "user", "content": prompt}],
            output_format=response_format,
        )
        return response.parsed_output
