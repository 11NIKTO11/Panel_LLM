from typing import Any, Dict, Union

from utils.llm import AnthropicClient, GeminiClient, OpenAIClient, OpenRouterClient

from .spec import short_name

_KNOWN = (OpenAIClient, AnthropicClient, GeminiClient, OpenRouterClient)


def build_registry(extra: Union[Dict[str, Any], None] = None) -> Dict[str, Any]:
    """
    Name -> object map for resolving a serialized RunSpec.

    Args:
        extra: everything not known here - in practice the response models, which are
            generated per topic by `make_response_models`. Descriptions need no entry: a
            spec names a column, not a callable.
    """
    registry = {}
    for obj in _KNOWN:
        registry[obj.__name__] = obj
        registry[short_name(obj)] = obj
    registry.update(extra or {})
    return registry
