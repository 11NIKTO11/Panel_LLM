"""
The three ways a call fails, in this package's own terms.

Every SDK names its failures differently, and one of them (OpenRouter over plain HTTP) has no
names at all. A client turns whatever its provider raised into one of these, so the batch
runner decides by kind and knows no SDK:

- `RateLimited` - wait, then try the call again.
- `CallFailed`  - try the call again: this attempt failed, the next may not.
- `BatchFailed` - stop the batch: every call would fail the same way.

Anything else a call raises is a bug or a failure no client foresaw, and stops the batch too.
"""


class LLMError(Exception):
    pass


class RateLimited(LLMError):
    pass


class CallFailed(LLMError):
    """A passing failure (network, timeout, the provider's own error) or an answer that cannot be used (not the schema, refused, cut off)."""


class BatchFailed(LLMError):
    """What no retry mends: a wrong key, no credit, an unknown model, a request the provider rejects as such."""


RETRYABLE_STATUS = (408, 409)


def status_error(status: int, message: str) -> LLMError:
    """The failure an HTTP status stands for: 429 is a rate limit, 408, 409 and 5xx pass, any other stops the batch."""
    if status == 429:
        return RateLimited(message)
    if status in RETRYABLE_STATUS or status >= 500:
        return CallFailed(message)
    return BatchFailed(message)
