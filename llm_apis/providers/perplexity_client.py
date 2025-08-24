"""
Perplexity client implementation.
"""

import time
from typing import Dict, Any, List, Optional, AsyncIterator
import httpx
import json

from ..base import BaseLLMClient
from ..models.response import ChatResponse, EmbeddingResponse, Usage, ChatChoice
from ..exceptions import (
    AuthenticationError,
    RateLimitError,
    InvalidRequestError,
    ModelNotFoundError,
    QuotaExceededError
)
from ..utils import config, create_retry_decorator
import logging

logger = logging.getLogger(__name__)


class PerplexityClient(BaseLLMClient):
    """Perplexity API client implementation."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize Perplexity client.
        
        Args:
            api_key: Perplexity API key
            **kwargs: Additional configuration options
        """
        super().__init__(api_key, **kwargs)
        
        # Get configuration
        provider_config = config.get_provider_config('perplexity')
        
        # Use provided API key or get from config
        self.api_key = api_key or provider_config.get('api_key')
        if not self.api_key:
            raise AuthenticationError("Perplexity API key is required")
        
        # Initialize HTTP client
        self.base_url = kwargs.get('base_url', provider_config.get('base_url'))
        self.timeout = kwargs.get('timeout', provider_config.get('timeout', 30))
        self.max_retries = kwargs.get('max_retries', provider_config.get('max_retries', 3))
        
        self.provider = "perplexity"
    
    @create_retry_decorator()
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        **kwargs
    ) -> ChatResponse:
        """
        Generate a chat completion using Perplexity API.
        
        Args:
            messages: List of message dictionaries
            model: Perplexity model name
            **kwargs: Additional parameters
            
        Returns:
            ChatResponse object
        """
        try:
            logger.info(f"Making Perplexity chat completion request with model: {model}")
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model,
                "messages": messages,
                "temperature": kwargs.get('temperature', 0.7),
                "max_tokens": kwargs.get('max_tokens'),
                "top_p": kwargs.get('top_p', 1.0),
                "frequency_penalty": kwargs.get('frequency_penalty', 0.0),
                "presence_penalty": kwargs.get('presence_penalty', 0.0),
                "stop": kwargs.get('stop'),
                "stream": False
            }
            
            # Remove None values
            payload = {k: v for k, v in payload.items() if v is not None}
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code != 200:
                    raise Exception(f"HTTP {response.status_code}: {response.text}")
                
                response_data = response.json()
                return self._convert_response(response_data)
            
        except Exception as e:
            logger.error(f"Perplexity API error: {str(e)}")
            raise self._handle_api_error(e)
    
    @create_retry_decorator()
    async def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        **kwargs
    ) -> AsyncIterator[ChatResponse]:
        """
        Generate a streaming chat completion using Perplexity API.
        
        Args:
            messages: List of message dictionaries
            model: Perplexity model name
            **kwargs: Additional parameters
            
        Yields:
            ChatResponse objects for each chunk
        """
        try:
            logger.info(f"Making Perplexity streaming chat completion request with model: {model}")
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model,
                "messages": messages,
                "temperature": kwargs.get('temperature', 0.7),
                "max_tokens": kwargs.get('max_tokens'),
                "top_p": kwargs.get('top_p', 1.0),
                "frequency_penalty": kwargs.get('frequency_penalty', 0.0),
                "presence_penalty": kwargs.get('presence_penalty', 0.0),
                "stop": kwargs.get('stop'),
                "stream": True
            }
            
            # Remove None values
            payload = {k: v for k, v in payload.items() if v is not None}
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        raise Exception(f"HTTP {response.status_code}: {error_text.decode()}")
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]  # Remove "data: " prefix
                            if data == "[DONE]":
                                break
                            
                            try:
                                chunk_data = json.loads(data)
                                yield self._convert_stream_chunk(chunk_data)
                            except json.JSONDecodeError:
                                continue
                
        except Exception as e:
            logger.error(f"Perplexity streaming API error: {str(e)}")
            raise self._handle_api_error(e)
    
    async def get_embeddings(
        self,
        texts: List[str],
        model: str,
        **kwargs
    ) -> EmbeddingResponse:
        """
        Perplexity doesn't support embeddings directly.
        This method raises NotImplementedError.
        """
        raise NotImplementedError("Perplexity does not provide embedding endpoints")
    
    def get_available_models(self) -> List[str]:
        """Get list of available Perplexity models."""
        return [
            "llama-3.1-sonar-small-128k-online",
            "llama-3.1-sonar-large-128k-online",
            "llama-3.1-sonar-huge-128k-online",
            "llama-3.1-8b-instruct",
            "llama-3.1-70b-instruct",
            "mistral-7b-instruct",
            "mixtral-8x7b-instruct"
        ]
    
    def _convert_response(self, response_data: Dict[str, Any]) -> ChatResponse:
        """Convert Perplexity response to ChatResponse."""
        choices = []
        for choice in response_data.get("choices", []):
            choices.append(ChatChoice(
                index=choice.get("index", 0),
                message=choice.get("message", {}),
                finish_reason=choice.get("finish_reason")
            ))
        
        usage = None
        if "usage" in response_data:
            usage_data = response_data["usage"]
            usage = Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0)
            )
        
        return ChatResponse(
            id=response_data.get("id", f"chatcmpl-{int(time.time())}"),
            created=response_data.get("created", int(time.time())),
            model=response_data.get("model", "unknown"),
            choices=choices,
            usage=usage,
            provider=self.provider
        )
    
    def _convert_stream_chunk(self, chunk_data: Dict[str, Any]) -> ChatResponse:
        """Convert Perplexity stream chunk to ChatResponse."""
        choices = []
        for choice in chunk_data.get("choices", []):
            delta = choice.get("delta", {})
            choices.append(ChatChoice(
                index=choice.get("index", 0),
                message={
                    "role": delta.get("role", "assistant"),
                    "content": delta.get("content", "")
                },
                finish_reason=choice.get("finish_reason")
            ))
        
        return ChatResponse(
            id=chunk_data.get("id", f"chatcmpl-{int(time.time())}"),
            created=chunk_data.get("created", int(time.time())),
            model=chunk_data.get("model", "unknown"),
            choices=choices,
            usage=None,  # Usage not available in streaming
            provider=self.provider
        )
    
    def _handle_api_error(self, error: Exception):
        """Handle and convert Perplexity API errors."""
        error_message = str(error)
        
        if "401" in error_message or "authentication" in error_message.lower():
            raise AuthenticationError(f"Perplexity authentication failed: {error_message}")
        elif "429" in error_message or "rate limit" in error_message.lower():
            raise RateLimitError(f"Perplexity rate limit exceeded: {error_message}")
        elif "400" in error_message or "invalid" in error_message.lower():
            raise InvalidRequestError(f"Invalid Perplexity request: {error_message}")
        elif "404" in error_message or "not found" in error_message.lower():
            raise ModelNotFoundError(f"Perplexity model not found: {error_message}")
        elif "quota" in error_message.lower():
            raise QuotaExceededError(f"Perplexity quota exceeded: {error_message}")
        else:
            raise Exception(f"Perplexity API error: {error_message}")