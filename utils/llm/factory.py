from typing import Union

from .anthropic_client import AnthropicClient
from .base import ANTHROPIC, GOOGLE, OPENAI, BaseLLMClient, detect_provider
from .gemini_client import GeminiClient
from .openai_client import OpenAIClient
from .openrouter_client import OpenRouterClient


def create_client(model: Union[str, None] = None, api_key: Union[str, None] = None) -> BaseLLMClient:
    """Pick the native provider client for a model name; fall back to OpenRouter."""
    if model:
        provider = detect_provider(model)
        if provider == ANTHROPIC:
            return AnthropicClient(api_key=api_key)
        if provider == GOOGLE:
            return GeminiClient(api_key=api_key)
        if provider == OPENAI:
            return OpenAIClient(api_key=api_key)
    return OpenRouterClient(api_key=api_key)
