import concurrent.futures
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Type, Union

import pandas as pd
from pydantic import BaseModel
from tqdm.auto import tqdm

from .base import BaseLLMClient
from .errors import CallFailed, RateLimited


@dataclass(frozen=True)
class Flaw:
    """
    What a caller's review found wrong with a parsed answer. A `usable` one may stand when no attempt
    does better; of two usable ones the smaller `distance` is the better.
    """
    reasons: Tuple[str, ...]
    usable: bool
    distance: float


Review = Callable[[BaseModel], Union[Flaw, None]]
FAILED, RATE_LIMITED = "failed", "rate_limited"


@dataclass(frozen=True)
class Batch:
    """
    What a batch gave, by row index.

    results: (prompt, parsed result).
    flawed: the reasons, for the results kept though flawed.
    attempts: one word per attempt that gave no accepted answer, in order - `failed`, `rate_limited`,
        or a flaw's reasons joined by `+`; only rows that had such an attempt. A row whose result is
        not flawed made one attempt more, the accepted one.
    skipped: the rows without a usable answer after every attempt.
    """
    results: Dict[int, Tuple[str, BaseModel]] = field(default_factory=dict)
    flawed: Dict[int, Tuple[str, ...]] = field(default_factory=dict)
    attempts: Dict[int, Tuple[str, ...]] = field(default_factory=dict)
    skipped: List[int] = field(default_factory=list)


class BatchStopped(Exception):
    """A batch that did not run to its end: `batch` is what came back before it stopped, `__cause__` what stopped it."""

    def __init__(self, batch: Batch):
        super().__init__(f"the batch stopped with {len(batch.results)} answers")
        self.batch = batch


def _process_single_item(
        i: int,
        row: pd.Series,
        prompt_creator: Callable[[pd.Series], str],
        model: str,
        response_format: Type[BaseModel],
        temperature: Union[float, None],
        client: BaseLLMClient,
        retries: int,
        instructions: Union[str, None],
        review: Union[Review, None],
) -> Tuple[int, Union[Tuple[str, BaseModel], None], Tuple[str, ...], Tuple[str, ...]]:
    """
    (row index, (prompt, parsed), the reasons of its flaw - none for an accepted answer, the attempts
    that gave no accepted answer as `Batch.attempts` words them), with None for the result when no
    attempt gave a usable one. `BatchFailed` and any error that is not one of this package's are
    not caught: they stop the batch.
    """
    prompt = prompt_creator(row)
    best: Union[Tuple[Flaw, BaseModel], None] = None
    tried: List[str] = []

    for attempt in range(retries):
        try:
            parsed = client.call_llm(model, prompt, response_format, temperature, instructions)
            if parsed is None:
                raise TypeError(f"{type(client).__name__}.call_llm returned None; a client raises CallFailed instead")
            flaw = review(parsed) if review else None
            if flaw is None:
                return i, (prompt, parsed), (), tuple(tried)
            tried.append("+".join(flaw.reasons))
            print(f"[Flawed] Row {i}: attempt {attempt + 1}/{retries}: {', '.join(flaw.reasons)}")
            if flaw.usable and (best is None or flaw.distance < best[0].distance):
                best = (flaw, parsed)

        except RateLimited:
            tried.append(RATE_LIMITED)
            wait = (2 ** attempt) + random.random()
            print(f"[RateLimit] Row {i}: retrying in {wait:.1f}s (attempt {attempt + 1}/{retries})")
            time.sleep(wait)

        except CallFailed as e:
            tried.append(FAILED)
            print(f"[CallFailed] Row {i}: attempt {attempt + 1}/{retries}: {str(e)[:200]}")

    if best:
        return i, (prompt, best[1]), best[0].reasons, tuple(tried)
    print(f"[GiveUp] Row {i}: exhausted {retries} attempts without a usable answer.")
    return i, None, (), tuple(tried)


def run_llm_batch(
        data: pd.DataFrame,
        *,
        prompt_creator: Callable[[pd.Series], str],
        response_model: Type[BaseModel],
        model: str,
        temperature: Union[float, None],
        client: BaseLLMClient,
        instructions: Union[str, None],
        review: Union[Review, None],
        max_workers: int,
        retries: int,
) -> Batch:
    """
    `instructions` is the system prompt, sent with every call; `review` judges each parsed answer,
    None accepting all. No argument has a default: the caller that owns the run states them all.

    Anything but `RateLimited` or `CallFailed` raised by a call, and an interrupt, stops the batch:
    calls not yet started are dropped, those under way finish and are kept, and `BatchStopped` is
    raised from what stopped it.
    """
    batch = Batch()

    def collect(future) -> None:
        i, parsed, reasons, tried = future.result()
        if tried:
            batch.attempts[i] = tried
        if parsed:
            batch.results[i] = parsed
            if reasons:
                batch.flawed[i] = reasons
        else:
            batch.skipped.append(i)

    print(f"Starting processing of {len(data)} rows with {max_workers} workers.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_process_single_item, i, row, prompt_creator, model, response_model, temperature, client, retries, instructions, review)
            for i, row in data.iterrows()
        ]
        collected = set()
        try:
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(data)):
                collected.add(future)
                collect(future)
        except BaseException as cause:
            executor.shutdown(wait=True, cancel_futures=True)
            for future in futures:
                if future not in collected and future.done() and not future.cancelled() and future.exception() is None:
                    collect(future)
            raise BatchStopped(batch) from cause

    return batch
