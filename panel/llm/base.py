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
    """
    One structured call per provider. `last_model` is the model id the provider reported for
    the most recent call, when it reports one, so a run record can pin an alias to a version.
    `TAKES_EFFORT` says whether the client is constructed with an `effort`.
    """
    last_model: Union[str, None] = None
    TAKES_EFFORT: bool = False

    def call_llm(
        self,
        model: str,
        prompt: str,
        response_format: Type[BaseModel],
        temperature: Union[float, None],
        instructions: Union[str, None] = None,
    ) -> BaseModel:
        """
        One structured call.

        Args:
            model: provider model id.
            prompt: the user message.
            response_format: Pydantic model the provider must return.
            temperature: sampling temperature, or None to send none.
            instructions: system prompt, if the caller frames the task with one.
        """
        raise NotImplementedError

    def schema_sent(self, model: str, response_format: Type[BaseModel]) -> dict:
        """The JSON schema a call for `model` sends for `response_format`: the model's own unless a client rewrites it."""
        return response_format.model_json_schema()
