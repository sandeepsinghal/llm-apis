"""
Custom exceptions for the LLM APIs package.
"""

from typing import Optional, Dict, Any


class LLMAPIError(Exception):
    """Base exception for all LLM API related errors."""
    
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response_data = response_data or {}


class AuthenticationError(LLMAPIError):
    """Raised when API authentication fails."""
    pass


class RateLimitError(LLMAPIError):
    """Raised when API rate limits are exceeded."""
    
    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class InvalidRequestError(LLMAPIError):
    """Raised when the request is invalid or malformed."""
    pass


class ModelNotFoundError(LLMAPIError):
    """Raised when the specified model is not available."""
    pass


class QuotaExceededError(LLMAPIError):
    """Raised when API quota is exceeded."""
    pass


class ServiceUnavailableError(LLMAPIError):
    """Raised when the API service is temporarily unavailable."""
    pass


class TimeoutError(LLMAPIError):
    """Raised when API request times out."""
    pass