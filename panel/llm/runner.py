import concurrent.futures
import random
import time
from typing import Callable, Dict, List, Tuple, Type, Union

import pandas as pd
from pydantic import BaseModel
from tqdm.auto import tqdm

from .base import BaseLLMClient
from .errors import CallFailed, RateLimited


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
) -> Tuple[int, Union[Tuple[str, BaseModel], None]]:
    """
    (row index, (prompt, parsed)), or (row index, None) when every attempt failed. `BatchFailed` and any
    error that is not one of this package's are not caught: they stop the batch.
    """
    prompt = prompt_creator(row)

    for attempt in range(retries):
        try:
            parsed = client.call_llm(model, prompt, response_format, temperature, instructions)
            if parsed is None:
                raise TypeError(f"{type(client).__name__}.call_llm returned None; a client raises CallFailed instead")
            return i, (prompt, parsed)

        except RateLimited:
            wait = (2 ** attempt) + random.random()
            print(f"[RateLimit] Row {i}: retrying in {wait:.1f}s (attempt {attempt + 1}/{retries})")
            time.sleep(wait)

        except CallFailed as e:
            print(f"[CallFailed] Row {i}: attempt {attempt + 1}/{retries}: {str(e)[:200]}")

    print(f"[GiveUp] Row {i}: exhausted {retries} attempts without valid parsed result.")
    return i, None


def run_llm_batch(
        data: pd.DataFrame,
        *,
        prompt_creator: Callable[[pd.Series], str],
        response_model: Type[BaseModel],
        model: str,
        temperature: Union[float, None],
        client: BaseLLMClient,
        instructions: Union[str, None],
        max_workers: int,
        retries: int,
) -> Tuple[Dict[int, Tuple[str, BaseModel]], List[int]]:
    """
    Row index -> (prompt, parsed result), and the indices of the rows skipped after `retries` attempts.
    `instructions` is the system prompt, sent with every call. No argument has a default: the caller
    that owns the run states them all.

    The first call to raise anything but `RateLimited` or `CallFailed` stops the batch: calls not yet
    started are dropped, those under way finish, and the error is raised.
    """
    results: Dict[int, Tuple[str, BaseModel]] = {}
    now_skipped: List[int] = []

    print(f"Starting processing of {len(data)} rows with {max_workers} workers.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_process_single_item, i, row, prompt_creator, model, response_model, temperature, client, retries, instructions)
            for i, row in data.iterrows()
        ]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(data)):
            try:
                i, parsed = future.result()
            except BaseException:
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            if parsed:
                results[i] = parsed
            else:
                now_skipped.append(i)

    return results, now_skipped
