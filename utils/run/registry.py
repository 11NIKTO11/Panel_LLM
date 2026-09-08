from typing import Any, Dict, Union

from utils.llm import AnthropicClient, GeminiClient, OpenAIClient, OpenRouterClient
from utils.voting import VotingPick, VotingProbabilities

from .spec import short_name

_KNOWN = (
    OpenAIClient, AnthropicClient, GeminiClient, OpenRouterClient,
    VotingProbabilities, VotingPick,
)


def build_registry(extra: Union[Dict[str, Any], None] = None) -> Dict[str, Any]:
    """
    Name -> object map for resolving a serialized RunSpec.

    Args:
        extra: entries defined outside `utils`, such as description functions.
    """
    registry = {}
    for obj in _KNOWN:
        registry[obj.__name__] = obj
        registry[short_name(obj)] = obj
    registry.update(extra or {})
    return registry
