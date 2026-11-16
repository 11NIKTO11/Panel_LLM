from typing import Iterable, Tuple, Type, Union

from pydantic import BaseModel

# Provider identifiers; also used as OpenRouter routing prefixes.
OPENAI = "openai"
ANTHROPIC = "anthropic"
GOOGLE = "google"

# Every effort level a provider names, lowest first; each model takes some of them.
EFFORT_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


def ordered_efforts(levels: Iterable[str]) -> Tuple[str, ...]:
    """Lowest first; a level outside `EFFORT_ORDER` raises, because its place among the others is unknown."""
    levels = list(levels)
    unknown = sorted(set(levels) - set(EFFORT_ORDER))
    if unknown:
        raise ValueError(f"effort levels {unknown} are not among {EFFORT_ORDER}")
    return tuple(level for level in EFFORT_ORDER if level in levels)


def detect_provider(model: str) -> str:
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
    `TAKES_EFFORT` says whether the client is constructed with an `effort`, the level every call sends.
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
        """`prompt` is the user message, `instructions` the system prompt; `temperature=None` sends none."""
        raise NotImplementedError

    def efforts(self, model: str) -> Union[Tuple[str, ...], None]:
        """
        The effort levels the provider reports for `model` today, lowest first: empty for a model that
        takes none, None when the provider has no such list to ask. A request to the provider, not a call of the model.
        """
        return None

    def parameters(self) -> dict:
        """What every call of this client sends or holds to beside `call_llm`'s arguments; a run record copies it."""
        return {}

    def schema_sent(self, model: str, response_format: Type[BaseModel]) -> dict:
        """The JSON schema a call for `model` sends for `response_format`: the model's own unless a client rewrites it."""
        return response_format.model_json_schema()
