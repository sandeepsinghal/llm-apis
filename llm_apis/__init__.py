"""
LLM APIs - A unified wrapper for various LLM provider APIs.

This package provides a consistent interface for interacting with different
Large Language Model providers including OpenAI, Anthropic, Perplexity, and Ollama.
"""

from .providers import (
    OpenAIClient,
    AnthropicClient,
    PerplexityClient,
    OllamaClient
)
from .base import BaseLLMClient
from .exceptions import (
    LLMAPIError,
    AuthenticationError,
    RateLimitError,
    InvalidRequestError
)

__version__ = "0.1.0"
__author__ = "LLM APIs Team"

__all__ = [
    "OpenAIClient",
    "AnthropicClient", 
    "PerplexityClient",
    "OllamaClient",
    "BaseLLMClient",
    "LLMAPIError",
    "AuthenticationError",
    "RateLimitError",
    "InvalidRequestError"
]