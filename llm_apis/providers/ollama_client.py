"""
Ollama client implementation.
"""

import time
from typing import Dict, Any, List, Optional, AsyncIterator
import httpx
import json

from ..base import BaseLLMClient
from ..models.response import ChatResponse, EmbeddingResponse, Usage, ChatChoice, EmbeddingData
from ..exceptions import (
    AuthenticationError,
    RateLimitError,
    InvalidRequestError,
    ModelNotFoundError,
    ServiceUnavailableError
)
from ..utils import config, create_retry_decorator
import logging

logger = logging.getLogger(__name__)


class OllamaClient(BaseLLMClient):
    """Ollama API client implementation."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize Ollama client.
        
        Args:
            api_key: Not used for Ollama (local deployment)
            **kwargs: Additional configuration options
        """
        super().__init__(api_key, **kwargs)
        
        # Get configuration
        provider_config = config.get_provider_config('ollama')
        
        # Initialize HTTP client
        self.base_url = kwargs.get('base_url', provider_config.get('base_url'))
        self.timeout = kwargs.get('timeout', provider_config.get('timeout', 30))
        self.max_retries = kwargs.get('max_retries', provider_config.get('max_retries', 3))
        
        self.provider = "ollama"
    
    @create_retry_decorator()
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        **kwargs
    ) -> ChatResponse:
        """
        Generate a chat completion using Ollama API.
        
        Args:
            messages: List of message dictionaries
            model: Ollama model name
            **kwargs: Additional parameters
            
        Returns:
            ChatResponse object
        """
        try:
            logger.info(f"Making Ollama chat completion request with model: {model}")
            
            headers = {
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": kwargs.get('temperature', 0.7),
                    "top_p": kwargs.get('top_p', 1.0),
                    "num_predict": kwargs.get('max_tokens', -1),
                    "stop": kwargs.get('stop', [])
                }
            }
            
            # Remove empty stop sequences
            if not payload["options"]["stop"]:
                del payload["options"]["stop"]
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code != 200:
                    raise Exception(f"HTTP {response.status_code}: {response.text}")
                
                response_data = response.json()
                return self._convert_response(response_data, model)
            
        except Exception as e:
            logger.error(f"Ollama API error: {str(e)}")
            raise self._handle_api_error(e)
    
    @create_retry_decorator()
    async def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        **kwargs
    ) -> AsyncIterator[ChatResponse]:
        """
        Generate a streaming chat completion using Ollama API.
        
        Args:
            messages: List of message dictionaries
            model: Ollama model name
            **kwargs: Additional parameters
            
        Yields:
            ChatResponse objects for each chunk
        """
        try:
            logger.info(f"Making Ollama streaming chat completion request with model: {model}")
            
            headers = {
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": kwargs.get('temperature', 0.7),
                    "top_p": kwargs.get('top_p', 1.0),
                    "num_predict": kwargs.get('max_tokens', -1),
                    "stop": kwargs.get('stop', [])
                }
            }
            
            # Remove empty stop sequences
            if not payload["options"]["stop"]:
                del payload["options"]["stop"]
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        raise Exception(f"HTTP {response.status_code}: {error_text.decode()}")
                    
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                chunk_data = json.loads(line)
                                yield self._convert_stream_chunk(chunk_data, model)
                                
                                # Check if this is the final chunk
                                if chunk_data.get("done", False):
                                    break
                            except json.JSONDecodeError:
                                continue
                
        except Exception as e:
            logger.error(f"Ollama streaming API error: {str(e)}")
            raise self._handle_api_error(e)
    
    @create_retry_decorator()
    async def get_embeddings(
        self,
        texts: List[str],
        model: str,
        **kwargs
    ) -> EmbeddingResponse:
        """
        Get embeddings using Ollama API.
        
        Args:
            texts: List of texts to embed
            model: Ollama embedding model name
            **kwargs: Additional parameters
            
        Returns:
            EmbeddingResponse object
        """
        try:
            logger.info(f"Making Ollama embeddings request with model: {model}")
            
            headers = {
                "Content-Type": "application/json"
            }
            
            data = []
            for i, text in enumerate(texts):
                payload = {
                    "model": model,
                    "prompt": text
                }
                
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.base_url}/api/embeddings",
                        headers=headers,
                        json=payload
                    )
                    
                    if response.status_code != 200:
                        raise Exception(f"HTTP {response.status_code}: {response.text}")
                    
                    response_data = response.json()
                    data.append(EmbeddingData(
                        embedding=response_data.get("embedding", []),
                        index=i
                    ))
            
            return EmbeddingResponse(
                data=data,
                model=model,
                usage=None,  # Ollama doesn't provide usage info
                provider=self.provider
            )
            
        except Exception as e:
            logger.error(f"Ollama embeddings API error: {str(e)}")
            raise self._handle_api_error(e)
    
    async def get_available_models(self) -> List[str]:
        """Get list of available Ollama models."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                
                if response.status_code != 200:
                    logger.warning("Could not fetch Ollama models")
                    return []
                
                data = response.json()
                models = [model["name"] for model in data.get("models", [])]
                return models
                
        except Exception as e:
            logger.warning(f"Error fetching Ollama models: {e}")
            return []
    
    def _convert_response(self, response_data: Dict[str, Any], model: str) -> ChatResponse:
        """Convert Ollama response to ChatResponse."""
        choices = []
        
        message = response_data.get("message", {})
        choices.append(ChatChoice(
            index=0,
            message={
                "role": message.get("role", "assistant"),
                "content": message.get("content", "")
            },
            finish_reason="stop" if response_data.get("done", False) else None
        ))
        
        # Calculate approximate usage
        usage = None
        if "eval_count" in response_data or "prompt_eval_count" in response_data:
            prompt_tokens = response_data.get("prompt_eval_count", 0)
            completion_tokens = response_data.get("eval_count", 0)
            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        
        return ChatResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=model,
            choices=choices,
            usage=usage,
            provider=self.provider
        )
    
    def _convert_stream_chunk(self, chunk_data: Dict[str, Any], model: str) -> ChatResponse:
        """Convert Ollama stream chunk to ChatResponse."""
        choices = []
        
        message = chunk_data.get("message", {})
        finish_reason = "stop" if chunk_data.get("done", False) else None
        
        choices.append(ChatChoice(
            index=0,
            message={
                "role": message.get("role", "assistant"),
                "content": message.get("content", "")
            },
            finish_reason=finish_reason
        ))
        
        return ChatResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=model,
            choices=choices,
            usage=None,  # Usage not available in streaming
            provider=self.provider
        )
    
    def _handle_api_error(self, error: Exception):
        """Handle and convert Ollama API errors."""
        error_message = str(error)
        
        if "connection" in error_message.lower() or "refused" in error_message.lower():
            raise ServiceUnavailableError(f"Ollama service unavailable: {error_message}")
        elif "404" in error_message or "not found" in error_message.lower():
            raise ModelNotFoundError(f"Ollama model not found: {error_message}")
        elif "400" in error_message or "invalid" in error_message.lower():
            raise InvalidRequestError(f"Invalid Ollama request: {error_message}")
        else:
            raise Exception(f"Ollama API error: {error_message}")