import json
import os
from typing import Type, Union

from pydantic import BaseModel

from .base import BaseLLMClient

try:
    from google import genai
except Exception:  # pragma: no cover - optional dependency guard
    genai = None


class GeminiClient(BaseLLMClient):
    SHORT_NAME = "Gemini"

    def __init__(self, api_key: Union[str, None] = None):
        if genai is None:
            raise ImportError("google-genai package is required for Gemini models.")
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY must be set for Gemini models.")
        self.client = genai.Client(api_key=key)

    def call_llm(self, model: str, prompt: str, response_format: Type[BaseModel], temperature: Union[float, None],
                 instructions: Union[str, None] = None) -> BaseModel:
        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                **({} if temperature is None else {"temperature": temperature}),
                "response_mime_type": "application/json",
                "response_json_schema": response_format.model_json_schema(),
                **({"system_instruction": instructions} if instructions else {}),
            },
        )
        self.last_model = getattr(response, "model_version", None)
        parsed = json.loads(response.text)
        return response_format(**parsed)
