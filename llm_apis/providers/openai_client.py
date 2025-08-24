"""
OpenAI client implementation.
"""

import time
from typing import Dict, Any, List, Optional, AsyncIterator
import httpx
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from ..base import BaseLLMClient
from ..models.response import ChatResponse, EmbeddingResponse, Usage, ChatChoice, EmbeddingData
from ..exceptions import (
    AuthenticationError,
    RateLimitError,
    InvalidRequestError,
    ModelNotFoundError,
    QuotaExceededError
)
from ..utils import config, create_retry_decorator, sanitize_request_data
import logging

logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    """OpenAI API client implementation."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key
            **kwargs: Additional configuration options
        """
        super().__init__(api_key, **kwargs)
        
        # Get configuration
        provider_config = config.get_provider_config('openai')
        
        # Use provided API key or get from config
        self.api_key = api_key or provider_config.get('api_key')
        if not self.api_key:
            raise AuthenticationError("OpenAI API key is required")
        
        # Initialize client
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=kwargs.get('base_url', provider_config.get('base_url')),
            timeout=kwargs.get('timeout', provider_config.get('timeout', 30)),
            max_retries=kwargs.get('max_retries', provider_config.get('max_retries', 3))
        )
        
        self.provider = "openai"
    
    @create_retry_decorator()
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        **kwargs
    ) -> ChatResponse:
        """
        Generate a chat completion using OpenAI API.
        
        Args:
            messages: List of message dictionaries
            model: OpenAI model name
            **kwargs: Additional parameters
            
        Returns:
            ChatResponse object
        """
        try:
            logger.info(f"Making OpenAI chat completion request with model: {model}")
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens'),
                top_p=kwargs.get('top_p', 1.0),
                frequency_penalty=kwargs.get('frequency_penalty', 0.0),
                presence_penalty=kwargs.get('presence_penalty', 0.0),
                stop=kwargs.get('stop'),
                stream=False
            )
            
            return self._convert_response(response)
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise self._handle_api_error(e)
    
    @create_retry_decorator()
    async def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        **kwargs
    ) -> AsyncIterator[ChatResponse]:
        """
        Generate a streaming chat completion using OpenAI API.
        
        Args:
            messages: List of message dictionaries
            model: OpenAI model name
            **kwargs: Additional parameters
            
        Yields:
            ChatResponse objects for each chunk
        """
        try:
            logger.info(f"Making OpenAI streaming chat completion request with model: {model}")
            
            stream = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens'),
                top_p=kwargs.get('top_p', 1.0),
                frequency_penalty=kwargs.get('frequency_penalty', 0.0),
                presence_penalty=kwargs.get('presence_penalty', 0.0),
                stop=kwargs.get('stop'),
                stream=True
            )
            
            async for chunk in stream:
                yield self._convert_stream_chunk(chunk)
                
        except Exception as e:
            logger.error(f"OpenAI streaming API error: {str(e)}")
            raise self._handle_api_error(e)
    
    @create_retry_decorator()
    async def get_embeddings(
        self,
        texts: List[str],
        model: str = "text-embedding-ada-002",
        **kwargs
    ) -> EmbeddingResponse:
        """
        Get embeddings using OpenAI API.
        
        Args:
            texts: List of texts to embed
            model: OpenAI embedding model name
            **kwargs: Additional parameters
            
        Returns:
            EmbeddingResponse object
        """
        try:
            logger.info(f"Making OpenAI embeddings request with model: {model}")
            
            response = await self.client.embeddings.create(
                model=model,
                input=texts,
                encoding_format=kwargs.get('encoding_format', 'float'),
                dimensions=kwargs.get('dimensions')
            )
            
            return self._convert_embedding_response(response)
            
        except Exception as e:
            logger.error(f"OpenAI embeddings API error: {str(e)}")
            raise self._handle_api_error(e)
    
    def get_available_models(self) -> List[str]:
        """Get list of available OpenAI models."""
        return [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "text-embedding-ada-002",
            "text-embedding-3-small",
            "text-embedding-3-large"
        ]
    
    def _convert_response(self, response: ChatCompletion) -> ChatResponse:
        """Convert OpenAI response to ChatResponse."""
        choices = []
        for choice in response.choices:
            choices.append(ChatChoice(
                index=choice.index,
                message={
                    "role": choice.message.role,
                    "content": choice.message.content or ""
                },
                finish_reason=choice.finish_reason
            ))
        
        usage = None
        if response.usage:
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens or 0,
                total_tokens=response.usage.total_tokens
            )
        
        return ChatResponse(
            id=response.id,
            created=response.created,
            model=response.model,
            choices=choices,
            usage=usage,
            provider=self.provider
        )
    
    def _convert_stream_chunk(self, chunk: ChatCompletionChunk) -> ChatResponse:
        """Convert OpenAI stream chunk to ChatResponse."""
        choices = []
        for choice in chunk.choices:
            message_content = ""
            if choice.delta and choice.delta.content:
                message_content = choice.delta.content
            
            choices.append(ChatChoice(
                index=choice.index,
                message={
                    "role": "assistant",
                    "content": message_content
                },
                finish_reason=choice.finish_reason
            ))
        
        return ChatResponse(
            id=chunk.id,
            created=chunk.created,
            model=chunk.model,
            choices=choices,
            usage=None,  # Usage not available in streaming
            provider=self.provider
        )
    
    def _convert_embedding_response(self, response) -> EmbeddingResponse:
        """Convert OpenAI embedding response to EmbeddingResponse."""
        data = []
        for item in response.data:
            data.append(EmbeddingData(
                embedding=item.embedding,
                index=item.index
            ))
        
        usage = None
        if response.usage:
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=0,
                total_tokens=response.usage.total_tokens
            )
        
        return EmbeddingResponse(
            data=data,
            model=response.model,
            usage=usage,
            provider=self.provider
        )
    
    def _handle_api_error(self, error: Exception):
        """Handle and convert OpenAI API errors."""
        error_message = str(error)
        
        if "401" in error_message or "authentication" in error_message.lower():
            raise AuthenticationError(f"OpenAI authentication failed: {error_message}")
        elif "429" in error_message or "rate limit" in error_message.lower():
            raise RateLimitError(f"OpenAI rate limit exceeded: {error_message}")
        elif "400" in error_message or "invalid" in error_message.lower():
            raise InvalidRequestError(f"Invalid OpenAI request: {error_message}")
        elif "404" in error_message or "not found" in error_message.lower():
            raise ModelNotFoundError(f"OpenAI model not found: {error_message}")
        elif "quota" in error_message.lower():
            raise QuotaExceededError(f"OpenAI quota exceeded: {error_message}")
        else:
            raise Exception(f"OpenAI API error: {error_message}")