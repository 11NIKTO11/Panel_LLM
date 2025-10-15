import os
import time
import random
import json
import concurrent.futures
from typing import Any, Callable, Dict, List, Tuple, Type, Union
import pandas as pd
from tqdm.auto import tqdm
from pydantic import BaseModel, ValidationError
from openai import OpenAI, RateLimitError


# Assuming these are imported from your specific modules
# from your_module import SoD_Utils, VotingResult

class VotingProcessor:
    def __init__(self, api_key: Union[str, None] = None, client: Union[Any, None] = None):
        """
        Initializes the processor with an OpenAI client.

        Args:
            api_key: OpenAI API key. If None, tries to fetch from env OPENAI_API_KEY.
            client: An existing OpenAI (or instructor-patched) client instance.
                    If provided, api_key is ignored.
        """
        if client:
            self.client = client
        else:
            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError("API key must be provided or set in OPENAI_API_KEY environment variable.")
            # Assuming you are using a library that adds .responses.parse to OpenAI
            # e.g., 'instructor'. If using standard OpenAI, this initialization might change.
            self.client = OpenAI(api_key=key)

    def _process_single_respondent(
            self,
            i: int,
            respondent: pd.Series,
            prompt_creator: Callable[[pd.Series], str],
            model: str,
            temperature: float,
            response_format: Type[BaseModel],
            retries: int = 3,
    ) -> Tuple[int, Union[BaseModel, None]]:
        """
        Internal method to process a single row/respondent with retry logic.
        """
        # Construct the full prompt using the passed function and suffix
        prompt = prompt_creator(respondent)

        for attempt in range(retries):
            try:
                # Using the specific syntax from your snippet
                response = self.client.responses.parse(
                    model=model,
                    temperature=temperature,
                    input=prompt,
                    text_format=response_format
                )
                return i, response.output_parsed

            except RateLimitError:
                wait = (2 ** attempt) + random.random()
                print(f"[RateLimit] Respondent {i}: retrying in {wait:.1f}s (attempt {attempt + 1}/{retries})")
                time.sleep(wait)

            except ValidationError as e:
                # Pydantic validation error on the output
                details = e.errors()[:2] if hasattr(e, 'errors') else str(e)[:200]
                print(
                    f"[ParseError] Respondent {i}: Pydantic validation failed on attempt {attempt + 1}/{retries}. Retrying... Details: {details}")
                # proceed to next attempt

            except (ValueError, json.JSONDecodeError) as e:
                # Raw JSON parsing error
                print(
                    f"[ParseError] Respondent {i}: JSON/ValueError on attempt {attempt + 1}/{retries}: {str(e)[:200]} — Retrying...")
                # proceed to next attempt

            except Exception as e:
                # Other unexpected errors: inform and stop retrying for this respondent
                print(f"[Error] Respondent {i}: {e} — not retrying this error type.")
                return i, None

        print(f"[GiveUp] Respondent {i}: exhausted {retries} attempts without valid parsed result.")
        return i, None

    def run(
            self,
            data: pd.DataFrame,
            prompt_creator: Callable[[pd.Series], str],
            response_model: Type[BaseModel],
            model: str = "gpt-4.1-nano",
            temperature: float = 0.0,
            max_workers: int = 10,
            retries: int = 3,
    ) -> Tuple[Dict[int, BaseModel], List[int]]:
        """
        Runs the parallel processing on the specified subset of data.

        Args:
            data: The full pandas DataFrame.
            indices_to_process: List of indices from dataframe to process (e.g. skipped_respondents).
            prompt_creator: Function that takes a pd.Series and returns description string (e.g. SoD_Utils.create_respondent_description).
            prompt_question: String to append to the description (e.g. prompt_question variable).
            response_model: The Pydantic class used for structural parsing (e.g. VotingResult.VotingResult).
            model: OpenAI model name.
            temperature: LLM temperature.
            max_workers: Number of threads for parallel execution.
            retries: Number of retries per respondent.

        Returns:
            A tuple containing:
            1. Dictionary of successful results {index: PydanticModel}.
            2. List of indices that failed (skipped_respondents).
        """

        # Validations
        results: Dict[int, BaseModel] = {}
        now_skipped: List[int] = []

        print(f"Starting processing of {len(data)} respondents with {max_workers} workers.")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Prepare arguments for the internal method
            future_to_idx = {
                executor.submit(
                    self._process_single_respondent,
                    i,
                    row,
                    prompt_creator,
                    model,
                    temperature,
                    response_model,
                    retries
                ): i
                for i, row in data.iterrows()
            }

            # Process as they complete
            for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(data)):
                i, voted = future.result()
                if voted:
                    results[i] = voted
                else:
                    now_skipped.append(i)

        return results, now_skipped