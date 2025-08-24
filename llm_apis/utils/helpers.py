"""
Helper utilities for LLM APIs.
"""

import time
import json
import logging
from typing import Dict, Any, Optional, List, Callable, TypeVar, Union
from functools import wraps
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from ..exceptions import RateLimitError, ServiceUnavailableError, TimeoutError

T = TypeVar('T')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def sanitize_request_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize request data by removing sensitive information.
    
    Args:
        data: Request data dictionary
        
    Returns:
        Sanitized data dictionary
    """
    sanitized = data.copy()
    
    # Remove sensitive keys
    sensitive_keys = ['api_key', 'authorization', 'password', 'secret']
    for key in sensitive_keys:
        if key in sanitized:
            sanitized[key] = "***REDACTED***"
    
    return sanitized


def calculate_token_estimate(text: str) -> int:
    """
    Rough estimation of token count for text.
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    # Rough approximation: 1 token ≈ 4 characters for English text
    return len(text) // 4


def create_retry_decorator(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0
) -> Callable:
    """
    Create a retry decorator for API calls.
    
    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time between retries
        max_wait: Maximum wait time between retries
        
    Returns:
        Retry decorator
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type((RateLimitError, ServiceUnavailableError, TimeoutError)),
        reraise=True
    )


def rate_limit_handler(retry_after: Optional[int] = None):
    """
    Handle rate limiting by waiting before retry.
    
    Args:
        retry_after: Seconds to wait before retry
    """
    wait_time = retry_after if retry_after else 1
    logger.warning(f"Rate limit hit, waiting {wait_time} seconds")
    time.sleep(wait_time)


def format_messages_for_provider(
    messages: List[Dict[str, str]], 
    provider: str
) -> List[Dict[str, str]]:
    """
    Format messages according to provider-specific requirements.
    
    Args:
        messages: List of message dictionaries
        provider: Provider name
        
    Returns:
        Formatted messages
    """
    formatted_messages = []
    
    for message in messages:
        formatted_message = message.copy()
        
        # Provider-specific formatting
        if provider == "anthropic":
            # Anthropic uses different role names
            if formatted_message.get("role") == "system":
                # Anthropic handles system messages differently
                formatted_message["role"] = "user"
                formatted_message["content"] = f"System: {formatted_message['content']}"
        
        formatted_messages.append(formatted_message)
    
    return formatted_messages


def extract_content_from_response(response_data: Dict[str, Any], provider: str) -> str:
    """
    Extract text content from provider response.
    
    Args:
        response_data: Raw response data
        provider: Provider name
        
    Returns:
        Extracted content string
    """
    try:
        if provider == "openai":
            return response_data["choices"][0]["message"]["content"]
        elif provider == "anthropic":
            return response_data["content"][0]["text"]
        elif provider == "perplexity":
            return response_data["choices"][0]["message"]["content"]
        elif provider == "ollama":
            return response_data["message"]["content"]
        else:
            # Generic fallback
            return str(response_data)
    except (KeyError, IndexError, TypeError) as e:
        logger.error(f"Failed to extract content from {provider} response: {e}")
        return ""


def validate_model_name(model: str, provider: str) -> bool:
    """
    Validate model name for a specific provider.
    
    Args:
        model: Model name
        provider: Provider name
        
    Returns:
        True if model name is valid
    """
    # Basic validation - can be extended with actual model lists
    if not model or not isinstance(model, str):
        return False
    
    # Provider-specific validation
    valid_prefixes = {
        "openai": ["gpt-", "text-", "davinci", "curie", "babbage", "ada"],
        "anthropic": ["claude-"],
        "perplexity": ["llama-", "mistral-", "sonar-"],
        "ollama": []  # Ollama accepts any model name
    }
    
    if provider in valid_prefixes and valid_prefixes[provider]:
        return any(model.startswith(prefix) for prefix in valid_prefixes[provider])
    
    return True  # Default to valid for unknown providers