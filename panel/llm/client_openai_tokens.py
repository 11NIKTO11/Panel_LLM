"""
An answer read off the model's token distribution instead of written by the model.

The prompt asks for one token out of a few; the provider reports, for each token it generated,
the candidates it weighed and how likely each was. The probabilities of the answer tokens at
the first position that offers any of them are the answer: continuous, where a model asked to
write probabilities writes round ones. No schema is sent.

A position is looked for rather than assumed to be the first, because a model that opens with a
quote or a space would otherwise read as having answered nothing. Tokens are matched with
decoration and case stripped, and candidates that decode to one answer are added up: `A`, ` A`
and `"A` are one answer. Nothing is rescaled - what the answers lack to 1 went to tokens that
are no answer, and is the caller's to judge.

Only the `TOP` likeliest candidates are reported, so an answer below them reads as 0.
Reasoning models report no token probabilities; a call of one stops the batch.
"""
import math
import os
from typing import Dict, List, Sequence, Tuple, Type, Union

from pydantic import BaseModel, ValidationError

from .base import BaseLLMClient
from .errors import BatchFailed, CallFailed, RateLimited, status_error

try:
    import openai
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency guard
    openai = OpenAI = None

TOP = 20  # the provider's cap
OUTPUT_TOKENS = 16  # the provider's floor
STRIP = " \t\n\"'*.)"
LOGPROBS = "message.output_text.logprobs"

Position = List[Tuple[str, float]]


def positions(response) -> List[Position]:
    """Per generated token, its candidates as (token, logprob)."""
    return [
        [(candidate.token, candidate.logprob) for candidate in token.top_logprobs]
        for item in response.output if item.type == "message"
        for content in item.content if content.type == "output_text"
        for token in (content.logprobs or [])
    ]


def scan(found: Sequence[Position], answers: Sequence[str]) -> Union[Dict[str, float], None]:
    """{answer: probability} at the first position offering any of `answers`, only those offered; None if none does."""
    wanted = {answer.strip(STRIP).casefold(): answer for answer in answers}
    for candidates in found:
        given: Dict[str, float] = {}
        for token, logprob in candidates:
            answer = wanted.get(token.strip(STRIP).casefold())
            if answer is not None:
                given[answer] = given.get(answer, 0.0) + math.exp(logprob)
        if given:
            return given
    return None


class OpenAITokensClient(BaseLLMClient):
    SHORT_NAME = "OpenAITokens"
    READS_TOKENS = True

    def __init__(self, api_key: Union[str, None] = None):
        if OpenAI is None:
            raise ImportError("openai package is required for OpenAI models.")
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY must be set for OpenAI models.")
        self.client = OpenAI(api_key=key)

    def parameters(self) -> dict:
        return {"top_logprobs": TOP, "max_output_tokens": OUTPUT_TOKENS, "include": [LOGPROBS], "stripped": STRIP}

    def schema_sent(self, model: str, response_format: Type[BaseModel]) -> None:
        return None

    def call_llm(self, model: str, prompt: str, response_format: Type[BaseModel], temperature: Union[float, None],
                 instructions: Union[str, None] = None) -> BaseModel:
        try:
            response = self.client.responses.create(
                model=model,
                input=prompt,
                top_logprobs=TOP,
                include=[LOGPROBS],
                max_output_tokens=OUTPUT_TOKENS,
                **({} if temperature is None else {"temperature": temperature}),
                **({"instructions": instructions} if instructions else {}),
            )
        except openai.RateLimitError as error:
            raise RateLimited(str(error)) from error
        except openai.APIStatusError as error:
            raise status_error(error.status_code, str(error)) from error
        except openai.OpenAIError as error:
            raise CallFailed(f"{type(error).__name__}: {error}") from error
        self.last_model = getattr(response, "model", None)
        found = positions(response)
        if not found:
            raise BatchFailed(f"'{model}' reported no token probabilities")
        given = scan(found, response_format.ANSWERS)
        if given is None:
            raise CallFailed(f"none of {list(response_format.ANSWERS)} among the candidates of the first {len(found)} tokens")
        try:
            return response_format.from_tokens(given)
        except ValidationError as error:
            raise CallFailed(f"{type(error).__name__}: {error}") from error
