"""
Anthropic client implementation.
"""

import time
from typing import Dict, Any, List, Optional, AsyncIterator
import httpx
from anthropic import AsyncAnthropic

from ..base import BaseLLMClient
from ..models.response import ChatResponse, EmbeddingResponse, Usage, ChatChoice
from ..exceptions import (
    AuthenticationError,
    RateLimitError,
    InvalidRequestError,
    ModelNotFoundError,
    QuotaExceededError
)
from ..utils import config, create_retry_decorator, format_messages_for_provider
import logging

logger = logging.getLogger(__name__)


class AnthropicClient(BaseLLMClient):
    """Anthropic API client implementation."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize Anthropic client.
        
        Args:
            api_key: Anthropic API key
            **kwargs: Additional configuration options
        """
        super().__init__(api_key, **kwargs)
        
        # Get configuration
        provider_config = config.get_provider_config('anthropic')
        
        # Use provided API key or get from config
        self.api_key = api_key or provider_config.get('api_key')
        if not self.api_key:
            raise AuthenticationError("Anthropic API key is required")
        
        # Initialize client
        self.client = AsyncAnthropic(
            api_key=self.api_key,
            base_url=kwargs.get('base_url', provider_config.get('base_url')),
            timeout=kwargs.get('timeout', provider_config.get('timeout', 30)),
            max_retries=kwargs.get('max_retries', provider_config.get('max_retries', 3))
        )
        
        self.provider = "anthropic"
    
    @create_retry_decorator()
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        **kwargs
    ) -> ChatResponse:
        """
        Generate a chat completion using Anthropic API.
        
        Args:
            messages: List of message dictionaries
            model: Anthropic model name
            **kwargs: Additional parameters
            
        Returns:
            ChatResponse object
        """
        try:
            logger.info(f"Making Anthropic chat completion request with model: {model}")
            
            # Format messages for Anthropic
            formatted_messages = format_messages_for_provider(messages, "anthropic")
            
            # Extract system message if present
            system_message = None
            user_messages = []
            
            for msg in formatted_messages:
                if msg.get("role") == "system":
                    system_message = msg["content"]
                else:
                    user_messages.append(msg)
            
            response = await self.client.messages.create(
                model=model,
                messages=user_messages,
                system=system_message,
                max_tokens=kwargs.get('max_tokens', 1024),
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 1.0),
                stop_sequences=kwargs.get('stop')
            )
            
            return self._convert_response(response, model)
            
        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise self._handle_api_error(e)
    
    @create_retry_decorator()
    async def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        **kwargs
    ) -> AsyncIterator[ChatResponse]:
        """
        Generate a streaming chat completion using Anthropic API.
        
        Args:
            messages: List of message dictionaries
            model: Anthropic model name
            **kwargs: Additional parameters
            
        Yields:
            ChatResponse objects for each chunk
        """
        try:
            logger.info(f"Making Anthropic streaming chat completion request with model: {model}")
            
            # Format messages for Anthropic
            formatted_messages = format_messages_for_provider(messages, "anthropic")
            
            # Extract system message if present
            system_message = None
            user_messages = []
            
            for msg in formatted_messages:
                if msg.get("role") == "system":
                    system_message = msg["content"]
                else:
                    user_messages.append(msg)
            
            stream = await self.client.messages.create(
                model=model,
                messages=user_messages,
                system=system_message,
                max_tokens=kwargs.get('max_tokens', 1024),
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 1.0),
                stop_sequences=kwargs.get('stop'),
                stream=True
            )
            
            async for chunk in stream:
                yield self._convert_stream_chunk(chunk, model)
                
        except Exception as e:
            logger.error(f"Anthropic streaming API error: {str(e)}")
            raise self._handle_api_error(e)
    
    async def get_embeddings(
        self,
        texts: List[str],
        model: str,
        **kwargs
    ) -> EmbeddingResponse:
        """
        Anthropic doesn't support embeddings directly.
        This method raises NotImplementedError.
        """
        raise NotImplementedError("Anthropic does not provide embedding endpoints")
    
    def get_available_models(self) -> List[str]:
        """Get list of available Anthropic models."""
        return [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2"
        ]
    
    def _convert_response(self, response, model: str) -> ChatResponse:
        """Convert Anthropic response to ChatResponse."""
        choices = []
        
        content = ""
        if response.content and len(response.content) > 0:
            content = response.content[0].text
        
        choices.append(ChatChoice(
            index=0,
            message={
                "role": "assistant",
                "content": content
            },
            finish_reason=response.stop_reason
        ))
        
        usage = None
        if hasattr(response, 'usage') and response.usage:
            usage = Usage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens
            )
        
        return ChatResponse(
            id=response.id,
            created=int(time.time()),
            model=model,
            choices=choices,
            usage=usage,
            provider=self.provider
        )
    
    def _convert_stream_chunk(self, chunk, model: str) -> ChatResponse:
        """Convert Anthropic stream chunk to ChatResponse."""
        choices = []
        
        content = ""
        finish_reason = None
        
        if hasattr(chunk, 'delta') and chunk.delta:
            if hasattr(chunk.delta, 'text'):
                content = chunk.delta.text
        
        if hasattr(chunk, 'type') and chunk.type == 'message_stop':
            finish_reason = "stop"
        
        choices.append(ChatChoice(
            index=0,
            message={
                "role": "assistant",
                "content": content
            },
            finish_reason=finish_reason
        ))
        
        return ChatResponse(
            id=getattr(chunk, 'id', f"chatcmpl-{int(time.time())}"),
            created=int(time.time()),
            model=model,
            choices=choices,
            usage=None,  # Usage not available in streaming
            provider=self.provider
        )
    
    def _handle_api_error(self, error: Exception):
        """Handle and convert Anthropic API errors."""
        error_message = str(error)
        
        if "401" in error_message or "authentication" in error_message.lower():
            raise AuthenticationError(f"Anthropic authentication failed: {error_message}")
        elif "429" in error_message or "rate limit" in error_message.lower():
            raise RateLimitError(f"Anthropic rate limit exceeded: {error_message}")
        elif "400" in error_message or "invalid" in error_message.lower():
            raise InvalidRequestError(f"Invalid Anthropic request: {error_message}")
        elif "404" in error_message or "not found" in error_message.lower():
            raise ModelNotFoundError(f"Anthropic model not found: {error_message}")
        elif "quota" in error_message.lower():
            raise QuotaExceededError(f"Anthropic quota exceeded: {error_message}")
        else:
            raise Exception(f"Anthropic API error: {error_message}")