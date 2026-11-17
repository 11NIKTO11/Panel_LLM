"""
A client that answers offline and deterministically, so the run layer works without a provider.

Every call derives its numbers from a digest of (model, temperature, prompt), so one prompt
always gets one answer and two runs of one spec produce identical files. `fail_calls` makes the
first n calls for each prompt raise, which is what exercises the batch runner's retries and the
executor's rerun rounds.

`FakeTokensClient` is the same for a framing with an answer menu: it shares the digest out over
the menu's tokens, a little of it left off the menu, as a token-reading client finds it.
"""
import hashlib
import threading
from collections import Counter
from typing import Type, Union

from pydantic import BaseModel

from panel.llm import BaseLLMClient, CallFailed


class FakeClient(BaseLLMClient):
    SHORT_NAME = "Fake"

    def __init__(self, api_key: Union[str, None] = None, fail_calls: int = 0):
        """`api_key` is ignored; present so every client constructs the same way."""
        self.fail_calls = fail_calls
        self._seen: Counter = Counter()
        self._lock = threading.Lock()

    def _attempt(self, prompt: str) -> None:
        with self._lock:
            self._seen[prompt] += 1
            attempt = self._seen[prompt]
        if attempt <= self.fail_calls:
            raise CallFailed(f"fake failure {attempt} of {self.fail_calls}")

    def call_llm(
        self,
        model: str,
        prompt: str,
        response_format: Type[BaseModel],
        temperature: Union[float, None],
        instructions: Union[str, None] = None,
    ) -> BaseModel:
        self._attempt(prompt)

        labels = list(getattr(response_format, "LABELS", {}))
        if not labels:
            raise TypeError("FakeClient answers only the models make_response_models builds")
        digest = hashlib.sha256(f"{model}|{temperature}|{prompt}".encode("utf-8")).digest()
        fields = response_format.model_fields
        payload = {}
        if "voted_or_not" in fields:
            voted = 0.5 + (digest[0] % 50) / 100
            payload["voted_or_not"] = {"voted": voted, "not_voted": round(1 - voted, 10)}
        if "options" in fields:
            weights = [1 + digest[1 + i % 30] for i in range(len(labels))]
            total = sum(weights)
            payload["options"] = [
                {"name": label, "probability": weight / total} for label, weight in zip(labels, weights)
            ]
        if "choice" in fields:
            payload["choice"] = labels[digest[31] % len(labels)]
        self.last_model = f"{model}-fake"
        return response_format.model_validate(payload)


class FakeTokensClient(FakeClient):
    SHORT_NAME = "FakeTokens"
    READS_TOKENS = True

    def call_llm(
        self,
        model: str,
        prompt: str,
        response_format: Type[BaseModel],
        temperature: Union[float, None],
        instructions: Union[str, None] = None,
    ) -> BaseModel:
        self._attempt(prompt)
        digest = hashlib.sha256(f"{model}|{temperature}|{prompt}".encode("utf-8")).digest()
        weights = [1 + digest[i % 32] for i in range(len(response_format.ANSWERS))]
        total = sum(weights) / (1 - digest[31] / 25600)
        self.last_model = f"{model}-fake"
        return response_format.from_tokens({token: weight / total for token, weight in zip(response_format.ANSWERS, weights)})

    def schema_sent(self, model: str, response_format: Type[BaseModel]) -> None:
        return None
