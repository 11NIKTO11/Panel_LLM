import concurrent.futures
import json
import random
import time
from typing import Callable, Dict, List, Tuple, Type, Union

import pandas as pd
from pydantic import BaseModel, ValidationError
from tqdm.auto import tqdm

from .base import BaseLLMClient
from .factory import create_client

try:
    from openai import RateLimitError
except Exception:  # pragma: no cover - optional dependency guard
    class RateLimitError(Exception):
        """Stand-in when the openai package is missing; never raised by other clients."""


def _process_single_item(
        i: int,
        row: pd.Series,
        prompt_creator: Callable[[pd.Series], str],
        model: str,
        response_format: Type[BaseModel],
        temperature: float,
        client: BaseLLMClient,
        retries: int = 3,
) -> Tuple[int, Union[Tuple[str, BaseModel], None]]:
    """
    Process a single row with retry logic and return prompt with result.
    """
    prompt = prompt_creator(row)

    for attempt in range(retries):
        try:
            parsed = client.call_llm(model, prompt, response_format, temperature)
            return i, (prompt, parsed)

        except RateLimitError:
            wait = (2 ** attempt) + random.random()
            print(f"[RateLimit] Row {i}: retrying in {wait:.1f}s (attempt {attempt + 1}/{retries})")
            time.sleep(wait)

        except ValidationError as e:
            details = e.errors()[:2] if hasattr(e, 'errors') else str(e)[:200]
            print(
                f"[ParseError] Row {i}: Pydantic validation failed on attempt {attempt + 1}/{retries}. Retrying... Details: {details}")

        except (ValueError, json.JSONDecodeError) as e:
            print(
                f"[ParseError] Row {i}: JSON/ValueError on attempt {attempt + 1}/{retries}: {str(e)[:200]} — Retrying...")

        except Exception as e:
            print(f"[Error] Row {i}: {e} — not retrying this error type.")
            return i, None

    print(f"[GiveUp] Row {i}: exhausted {retries} attempts without valid parsed result.")
    return i, None


def run_llm_batch(
        data: pd.DataFrame,
        prompt_creator: Callable[[pd.Series], str],
        response_model: Type[BaseModel],
        model: str = "gpt-4.1-nano",
        temperature: float = 0.0,
        max_workers: int = 10,
        retries: int = 3,
        client: Union[BaseLLMClient, None] = None,
        api_key: Union[str, None] = None,
) -> Tuple[Dict[int, Tuple[str, BaseModel]], List[int]]:
    """
    Run parallel LLM calls across DataFrame rows, selecting the right client by model unless provided explicitly.
    Returns a dict of row index -> (prompt, parsed_result) and a list of skipped row indices.
    """
    results: Dict[int, Tuple[str, BaseModel]] = {}
    now_skipped: List[int] = []
    llm_client = client or create_client(model, api_key=api_key)

    print(f"Starting processing of {len(data)} rows with {max_workers} workers.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                _process_single_item,
                i,
                row,
                prompt_creator,
                model,
                response_model,
                temperature,
                llm_client,
                retries
            ): i
            for i, row in data.iterrows()
        }

        for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(data)):
            i, parsed = future.result()
            if parsed:
                results[i] = parsed
            else:
                now_skipped.append(i)

    return results, now_skipped
