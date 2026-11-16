# Deliberately self-contained (no imports from the surrounding project, no import-time
# side effects) so the package can be copied between projects as a unit.

from .base import ANTHROPIC, EFFORT_ORDER, GOOGLE, OPENAI, BaseLLMClient, detect_provider, ordered_efforts
from .client_openai import OpenAIClient
from .client_anthropic import AnthropicClient, AnthropicLegacyClient
from .client_gemini import GeminiClient
from .client_openrouter import OpenRouterClient
from .factory import create_client
from .runner import run_llm_batch
