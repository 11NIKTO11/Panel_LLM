from typing import Type, Union

from pydantic import BaseModel

# Provider identifiers; also used as OpenRouter routing prefixes.
OPENAI = "openai"
ANTHROPIC = "anthropic"
GOOGLE = "google"


def detect_provider(model: str) -> str:
    """Guess the provider from a bare model name; returns "" when unknown."""
    lower = model.lower()
    if "claude" in lower:
        return ANTHROPIC
    if "gemini" in lower:
        return GOOGLE
    if "gpt" in lower or "openai" in lower:
        return OPENAI
    return ""


class BaseLLMClient:
    def call_llm(
        self,
        model: str,
        prompt: str,
        response_format: Type[BaseModel],
        temperature: float,
        instructions: Union[str, None] = None,
    ) -> BaseModel:
        """
        One structured call.

        Args:
            model: provider model id.
            prompt: the user message.
            response_format: Pydantic model the provider must return.
            temperature: sampling temperature.
            instructions: system prompt, if the caller frames the task with one.
        """
        raise NotImplementedError
