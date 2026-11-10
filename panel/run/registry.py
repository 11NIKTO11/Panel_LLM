"""Name -> client class, so the client a spec or a record names resolves to code."""
from typing import Dict, Type

from panel.llm import AnthropicClient, AnthropicLegacyClient, BaseLLMClient, GeminiClient, OpenAIClient, OpenRouterClient

from .fake import FakeClient

KNOWN = (OpenAIClient, AnthropicClient, AnthropicLegacyClient, GeminiClient, OpenRouterClient, FakeClient)


def short_name(cls) -> str:
    """A class's SHORT_NAME if it declares one, else its __name__."""
    return getattr(cls, "SHORT_NAME", None) or cls.__name__


def registry() -> Dict[str, Type[BaseLLMClient]]:
    """Every known client under both its short name and its class name."""
    return {name: cls for cls in KNOWN for name in (short_name(cls), cls.__name__)}


def client_class(name: str) -> Type[BaseLLMClient]:
    """
    Raises:
        KeyError: naming the known clients, if `name` is not one of them.
    """
    known = registry()
    if name not in known:
        raise KeyError(f"unknown client '{name}'; known: {sorted(short_name(cls) for cls in KNOWN)}")
    return known[name]
