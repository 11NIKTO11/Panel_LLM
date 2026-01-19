import os
import time
import random
import json
import concurrent.futures
from typing import Any, Callable, Dict, List, Tuple, Type, Union

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from pydantic import BaseModel, ValidationError

try:
    from openai import OpenAI, RateLimitError
except Exception:  # pragma: no cover - optional dependency guard
    OpenAI = None
    RateLimitError = Exception

try:
    from anthropic import Anthropic
except Exception:  # pragma: no cover
    Anthropic = None

try:
    from google import genai
except Exception:  # pragma: no cover
    genai = None

def generate_prob_vectors_df(n, m):
    random_vectors = np.random.rand(n, m)
    probability_vectors = random_vectors / random_vectors.sum(axis=1, keepdims=True)
    df = pd.DataFrame(probability_vectors)
    return df

class BaseLLMClient:
    def call_llm(
        self,
        model: str,
        prompt: str,
        response_format: Type[BaseModel],
        temperature: float,
    ) -> BaseModel:
        raise NotImplementedError


class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: Union[str, None] = None):
        if OpenAI is None:
            raise ImportError("openai package is required for OpenAI models.")
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY must be set for OpenAI models.")
        self.client = OpenAI(api_key=key)

    def call_llm(self, model: str, prompt: str, response_format: Type[BaseModel], temperature: float) -> BaseModel:
        response = self.client.responses.parse(
            model=model,
            temperature=temperature,
            input=prompt,
            text_format=response_format
        )
        return response.output_parsed


class ClaudeClient(BaseLLMClient):
    def __init__(self, api_key: Union[str, None] = None):
        if Anthropic is None:
            raise ImportError("anthropic package is required for Claude models.")
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY must be set for Claude models.")
        self.client = Anthropic(api_key=key)

    def call_llm(self, model: str, prompt: str, response_format: Type[BaseModel], temperature: float) -> BaseModel:
        response = self.client.beta.messages.parse(
            model=model,
            betas=["structured-outputs-2025-11-13"],
            messages=[{"role": "user", "content": prompt}],
            output_format=response_format,
        )
        return getattr(response, "parsed_output", None)


class GeminiClient(BaseLLMClient):
    def __init__(self, api_key: Union[str, None] = None):
        if genai is None:
            raise ImportError("google-genai package is required for Gemini models.")
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY must be set for Gemini models.")
        self.client = genai.Client(api_key=key)

    def call_llm(self, model: str, prompt: str, response_format: Type[BaseModel], temperature: float) -> BaseModel:
        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": response_format.model_json_schema(),
            },
        )
        raw_text = getattr(response, "text", None)
        if raw_text is None and hasattr(response, "candidates"):
            try:
                raw_text = response.candidates[0].content.parts[0].text
            except Exception:
                raw_text = None
        if not raw_text:
            raise ValueError("Gemini response missing text payload.")
        parsed = json.loads(raw_text)
        return response_format(**parsed)


def _detect_provider(model: str) -> str:
    lower = model.lower()
    if "claude" in lower:
        return "claude"
    if "gemini" in lower:
        return "gemini"
    if "gpt" in lower or "openai" in lower:
        return "openai"
    return "openai"


def create_client(model: str, api_key: Union[str, None] = None) -> Union[BaseLLMClient,None]:
    provider = _detect_provider(model)
    if provider == "claude":
        return ClaudeClient(api_key=api_key)
    if provider == "gemini":
        return GeminiClient(api_key=api_key)
    if provider == "openai":
        return OpenAIClient(api_key=api_key)
    return None


def _process_single_respondent(
        i: int,
        respondent: pd.Series,
        prompt_creator: Callable[[pd.Series], str],
        model: str,
        response_format: Type[BaseModel],
        temperature: float,
        client: BaseLLMClient,
        retries: int = 3,
) -> Tuple[int, Union[BaseModel, None]]:
    """
    Process a single row/respondent with retry logic.
    """
    prompt = prompt_creator(respondent)

    for attempt in range(retries):
        try:
            parsed = client.call_llm(model, prompt, response_format, temperature)
            return i, parsed

        except RateLimitError:
            wait = (2 ** attempt) + random.random()
            print(f"[RateLimit] Respondent {i}: retrying in {wait:.1f}s (attempt {attempt + 1}/{retries})")
            time.sleep(wait)

        except ValidationError as e:
            details = e.errors()[:2] if hasattr(e, 'errors') else str(e)[:200]
            print(
                f"[ParseError] Respondent {i}: Pydantic validation failed on attempt {attempt + 1}/{retries}. Retrying... Details: {details}")

        except (ValueError, json.JSONDecodeError) as e:
            print(
                f"[ParseError] Respondent {i}: JSON/ValueError on attempt {attempt + 1}/{retries}: {str(e)[:200]} — Retrying...")

        except Exception as e:
            print(f"[Error] Respondent {i}: {e} — not retrying this error type.")
            return i, None

    print(f"[GiveUp] Respondent {i}: exhausted {retries} attempts without valid parsed result.")
    return i, None


def run_voting_simulation(
        data: pd.DataFrame,
        prompt_creator: Callable[[pd.Series], str],
        response_model: Type[BaseModel],
        model: str = "gpt-4.1-nano",
        temperature: float = 0.0,
        max_workers: int = 10,
        retries: int = 3,
        client: Union[BaseLLMClient, None] = None,
        api_key: Union[str, None] = None,
) -> Tuple[Dict[int, BaseModel], List[int]]:
    """
    Run parallel processing across respondents, selecting the right client by model unless provided explicitly.
    """
    results: Dict[int, BaseModel] = {}
    now_skipped: List[int] = []
    llm_client = client or create_client(model, api_key=api_key)

    print(f"Starting processing of {len(data)} respondents with {max_workers} workers.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                _process_single_respondent,
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
            i, voted = future.result()
            if voted:
                results[i] = voted
            else:
                now_skipped.append(i)

    return results, now_skipped
