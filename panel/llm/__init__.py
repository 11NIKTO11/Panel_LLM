# LLM client package: provider adapters, schema-compat fixups, client factory,
# and a parallel batch runner. Deliberately self-contained (no imports from the
# surrounding project, no import-time side effects) so it can be copied between
# projects as a unit.

from .base import ANTHROPIC, GOOGLE, OPENAI, BaseLLMClient, detect_provider
from .client_openai import OpenAIClient
from .client_anthropic import AnthropicClient
from .client_gemini import GeminiClient
from .client_openrouter import OpenRouterClient
from .factory import create_client
from .runner import run_llm_batch
