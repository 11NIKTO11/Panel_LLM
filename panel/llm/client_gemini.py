import json
import os
from typing import Type, Union

from pydantic import BaseModel, ValidationError

from .base import BaseLLMClient
from .errors import CallFailed, status_error

try:
    from google import genai
    from google.genai import errors
except Exception:  # pragma: no cover - optional dependency guard
    genai = errors = None


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
        """A failure of the connection itself is not turned into `CallFailed`: the SDK names no error for it."""
        try:
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
        except errors.APIError as error:
            raise status_error(error.code, str(error)) from error
        self.last_model = getattr(response, "model_version", None)
        if response.text is None:
            raise CallFailed(f"no text from '{model}': blocked, or cut off before the answer")
        try:
            return response_format(**json.loads(response.text))
        except (ValueError, ValidationError) as error:
            raise CallFailed(f"answer of '{model}' is not the schema: {str(error)[:200]}") from error
