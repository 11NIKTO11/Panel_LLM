"""Name -> client class, so the client a spec or a record names resolves to code."""
from typing import Dict, Type

from panel.llm import AnthropicClient, AnthropicLegacyClient, BaseLLMClient, GeminiClient, OpenAIClient, OpenAITokensClient, OpenRouterClient

from .fake import FakeClient, FakeTokensClient

KNOWN = (OpenAIClient, OpenAITokensClient, AnthropicClient, AnthropicLegacyClient, GeminiClient, OpenRouterClient, FakeClient, FakeTokensClient)


def short_name(cls) -> str:
    return getattr(cls, "SHORT_NAME", None) or cls.__name__


def registry() -> Dict[str, Type[BaseLLMClient]]:
    return {name: cls for cls in KNOWN for name in (short_name(cls), cls.__name__)}


def client_class(name: str) -> Type[BaseLLMClient]:
    known = registry()
    if name not in known:
        raise KeyError(f"unknown client '{name}'; known: {sorted(short_name(cls) for cls in KNOWN)}")
    return known[name]
