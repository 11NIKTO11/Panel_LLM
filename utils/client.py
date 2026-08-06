import os
import time
import random
import requests
import json
import concurrent.futures
import os

import pandas as pd
import numpy as np

from dotenv import load_dotenv
from tqdm.auto import tqdm
from pydantic import BaseModel, ValidationError
from typing import Any, Callable, Dict, List, Tuple, Type, Union
from utils import constants

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

load_dotenv()  # reads .env and sets os.environ

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

class AnthropicClient(BaseLLMClient):
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
            max_tokens=2048,
            temperature=temperature,
            betas=["structured-outputs-2025-11-13"],
            messages=[{"role": "user", "content": prompt}],
            output_format=response_format,
        )
        return response.parsed_output

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
                "temperature": temperature,
                "response_mime_type": "application/json",
                "response_json_schema": response_format.model_json_schema(),
            },
        )
        parsed = json.loads(response.text)
        return response_format(**parsed)

class OpenRouterClient(BaseLLMClient):
    def __init__(self, api_key: Union[str, None] = None):
        if genai is None:
            raise ImportError("google-genai package is required for Gemini models.")
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY must be set for Gemini models.")
        self.header = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def call_llm(self, model: str, prompt: str, response_format: Type[BaseModel], temperature: float) -> BaseModel:
        provider = _detect_provider(model)
        response_format_json = response_format.model_json_schema()

        if provider == constants.OPENAI:
            response_format_json = _set_additional_properties_false(response_format_json)
            response_format_json = _move_ref_descriptions(response_format_json)
        elif provider == constants.ANTHROPIC:
            response_format_json = _set_additional_properties_false(response_format_json)
            response_format_json = _remove_min_max(response_format_json)

        response = requests.post(
            self.url,
            headers=self.header,
            json={
                "model": f"{provider}/{self._fix_model(model)}",
                "messages": [
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
                "response_format": {
                    "type": "json_schema",
                    "json_schema":{
                        "name": response_format.__name__,
                        "strict": True,
                        "schema": response_format_json,
                    }
                },
            },
        )
        data = response.json()
        try:
            text = data["choices"][0]["message"]["content"]
            parsed = json.loads(text)
            return response_format(**parsed)
        except Exception as e:
            raise ValueError(f"Failed to parse response: {data}")

    def _fix_model(self, model: str):
        if model == constants.CLAUDE_SONNET_45:
            model = model.replace("4-5", "4.5")
        return model

def _set_additional_properties_false(schema: dict) -> dict:
    """
    Recursively traverses a JSON schema and sets 'additionalProperties'
    to False for all objects.
    """
    if isinstance(schema, dict):
        if schema.get("type") == "object" and "properties" in schema:
            schema["additionalProperties"] = False
        for key, value in schema.items():
            schema[key] = _set_additional_properties_false(value)
    elif isinstance(schema, list):
        for i, item in enumerate(schema):
            schema[i] = _set_additional_properties_false(item)
    return schema

def _move_ref_descriptions(schema: dict) -> dict:
    """
    Finds descriptions next to $refs and moves them into the referenced definition.
    This is required for models like gpt-4.1-nano.
    """
    if isinstance(schema, dict):
        # The pattern to fix is a dictionary with both '$ref' and 'description'
        if '$ref' in schema and 'description' in schema:
            ref_path = schema['$ref']
            description = schema.pop('description') # Remove description from here

            # The path is typically '#/$defs/ModelName'
            try:
                parts = ref_path.strip('#/').split('/')
                target = schema
                # Find the root of the schema to navigate from
                # This is a simplification; a more robust solution might need to pass the root down.
                # For a typical Pydantic schema, this will work if called on the top-level dict.
                if '$defs' in schema:
                    target_def = schema['$defs'][parts[1]]
                    if 'description' not in target_def: # Don't overwrite existing description
                         target_def['description'] = description
                    else:
                        print(f"Warning: Description for ref {ref_path} already exists")
            except (KeyError, IndexError) as e:
                # Could not find the referenced definition, just leave it.
                print(f"Warning: Could not move description for ref {ref_path}: {e}")
                schema['description'] = description # Put it back if failed

        # Recurse through the rest of the schema
        for key, value in schema.items():
            schema[key] = _move_ref_descriptions(value)

    elif isinstance(schema, list):
        for i, item in enumerate(schema):
            schema[i] = _move_ref_descriptions(item)

    return schema

def _remove_min_max(schema: dict) -> dict:
    """
    Recursively removes 'minimum' and 'maximum' keys from a JSON schema.
    Required for models like claude-sonnet-4-5.
    """
    if isinstance(schema, dict):
        if schema.get("type") in ["number", "integer"]:
            schema.pop("minimum", None)
            schema.pop("maximum", None)

        for key, value in schema.items():
            schema[key] = _remove_min_max(value)

    elif isinstance(schema, list):
        for i, item in enumerate(schema):
            schema[i] = _remove_min_max(item)

    return schema

def _detect_provider(model: str) -> str:
    lower = model.lower()
    if "claude" in lower:
        return constants.ANTHROPIC
    if "gemini" in lower:
        return constants.GOOGLE
    if "gpt" in lower or "openai" in lower:
        return constants.OPENAI
    return ""

def create_client(model: Union[str, None] = None, api_key: Union[str, None] = None) -> Union[BaseLLMClient,None]:
    if model:
        provider = _detect_provider(model)
        if provider == constants.ANTHROPIC:
            return AnthropicClient(api_key=api_key)
        if provider == constants.GOOGLE:
            return GeminiClient(api_key=api_key)
        if provider == constants.OPENAI:
            return OpenAIClient(api_key=api_key)
    return OpenRouterClient(api_key=api_key)

def _process_single_respondent(
        i: int,
        respondent: pd.Series,
        prompt_creator: Callable[[pd.Series], str],
        model: str,
        response_format: Type[BaseModel],
        temperature: float,
        client: BaseLLMClient,
        retries: int = 3,
) -> Tuple[int, Union[Tuple[str, BaseModel], None]]:
    """
    Process a single row/respondent with retry logic and return prompt with result.
    """
    prompt = prompt_creator(respondent)

    for attempt in range(retries):
        try:
            parsed = client.call_llm(model, prompt, response_format, temperature)
            return i, (prompt, parsed)

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
) -> Tuple[Dict[int, Tuple[str, BaseModel]], List[int]]:
    """
    Run parallel processing across respondents, selecting the right client by model unless provided explicitly.
    Returns a dict of respondent_id -> (prompt, parsed_result).
    """
    results: Dict[int, Tuple[str, BaseModel]] = {}
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
