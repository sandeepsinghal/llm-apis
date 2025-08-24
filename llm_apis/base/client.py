"""
Base client abstract class for all LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, AsyncIterator, Iterator, Union
from pydantic import BaseModel
import asyncio
from ..models.request import ChatRequest, EmbeddingRequest
from ..models.response import ChatResponse, EmbeddingResponse


class BaseLLMClient(ABC):
    """
    Abstract base class for all LLM provider clients.
    
    This class defines the common interface that all provider-specific
    clients must implement, ensuring consistency across different providers.
    """
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize the base client.
        
        Args:
            api_key: API key for the provider
            **kwargs: Additional provider-specific configuration
        """
        self.api_key = api_key
        self.config = kwargs
        
    @abstractmethod
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        **kwargs
    ) -> ChatResponse:
        """
        Generate a chat completion response.
        
        Args:
            messages: List of message dictionaries
            model: Model identifier
            **kwargs: Additional parameters
            
        Returns:
            ChatResponse object
        """
        pass
        
    @abstractmethod
    async def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        **kwargs
    ) -> AsyncIterator[ChatResponse]:
        """
        Generate a streaming chat completion response.
        
        Args:
            messages: List of message dictionaries
            model: Model identifier
            **kwargs: Additional parameters
            
        Yields:
            ChatResponse objects for each chunk
        """
        pass
        
    @abstractmethod
    async def get_embeddings(
        self,
        texts: List[str],
        model: str,
        **kwargs
    ) -> EmbeddingResponse:
        """
        Get embeddings for the provided texts.
        
        Args:
            texts: List of texts to embed
            model: Model identifier
            **kwargs: Additional parameters
            
        Returns:
            EmbeddingResponse object
        """
        pass
        
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """
        Get list of available models for this provider.
        
        Returns:
            List of model identifiers
        """
        pass
        
    # Synchronous wrapper methods
    def chat_completion_sync(
        self,
        messages: List[Dict[str, str]],
        model: str,
        **kwargs
    ) -> ChatResponse:
        """Synchronous wrapper for chat_completion."""
        return asyncio.run(self.chat_completion(messages, model, **kwargs))
        
    def get_embeddings_sync(
        self,
        texts: List[str],
        model: str,
        **kwargs
    ) -> EmbeddingResponse:
        """Synchronous wrapper for get_embeddings."""
        return asyncio.run(self.get_embeddings(texts, model, **kwargs))