"""
Tests for OpenAI client.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from llm_apis.providers.openai_client import OpenAIClient
from llm_apis.exceptions import AuthenticationError, RateLimitError


class TestOpenAIClient:
    """Test cases for OpenAI client."""
    
    def test_init_without_api_key(self):
        """Test initialization without API key raises error."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(AuthenticationError):
                OpenAIClient()
    
    def test_init_with_api_key(self):
        """Test initialization with API key."""
        client = OpenAIClient(api_key="test-key")
        assert client.api_key == "test-key"
        assert client.provider == "openai"
    
    @pytest.mark.asyncio
    async def test_chat_completion_success(self):
        """Test successful chat completion."""
        with patch('llm_apis.providers.openai_client.AsyncOpenAI') as mock_openai:
            # Mock response
            mock_response = MagicMock()
            mock_response.id = "test-id"
            mock_response.created = 1234567890
            mock_response.model = "gpt-3.5-turbo"
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].index = 0
            mock_response.choices[0].message.role = "assistant"
            mock_response.choices[0].message.content = "Hello, world!"
            mock_response.choices[0].finish_reason = "stop"
            mock_response.usage = MagicMock()
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 5
            mock_response.usage.total_tokens = 15
            
            # Mock client
            mock_client_instance = AsyncMock()
            mock_client_instance.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client_instance
            
            # Test
            client = OpenAIClient(api_key="test-key")
            response = await client.chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-3.5-turbo"
            )
            
            assert response.id == "test-id"
            assert response.model == "gpt-3.5-turbo"
            assert response.choices[0].message["content"] == "Hello, world!"
            assert response.usage.total_tokens == 15
            assert response.provider == "openai"
    
    @pytest.mark.asyncio
    async def test_chat_completion_stream(self):
        """Test streaming chat completion."""
        with patch('llm_apis.providers.openai_client.AsyncOpenAI') as mock_openai:
            # Mock stream chunks
            chunk1 = MagicMock()
            chunk1.id = "test-id"
            chunk1.created = 1234567890
            chunk1.model = "gpt-3.5-turbo"
            chunk1.choices = [MagicMock()]
            chunk1.choices[0].index = 0
            chunk1.choices[0].delta.content = "Hello"
            chunk1.choices[0].finish_reason = None
            
            chunk2 = MagicMock()
            chunk2.id = "test-id"
            chunk2.created = 1234567890
            chunk2.model = "gpt-3.5-turbo"
            chunk2.choices = [MagicMock()]
            chunk2.choices[0].index = 0
            chunk2.choices[0].delta.content = " world!"
            chunk2.choices[0].finish_reason = "stop"
            
            # Mock async iterator
            async def mock_stream():
                yield chunk1
                yield chunk2
            
            # Mock client
            mock_client_instance = AsyncMock()
            mock_client_instance.chat.completions.create.return_value = mock_stream()
            mock_openai.return_value = mock_client_instance
            
            # Test
            client = OpenAIClient(api_key="test-key")
            chunks = []
            async for chunk in client.chat_completion_stream(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-3.5-turbo"
            ):
                chunks.append(chunk)
            
            assert len(chunks) == 2
            assert chunks[0].choices[0].message["content"] == "Hello"
            assert chunks[1].choices[0].message["content"] == " world!"
            assert chunks[1].choices[0].finish_reason == "stop"
    
    @pytest.mark.asyncio
    async def test_get_embeddings_success(self):
        """Test successful embeddings generation."""
        with patch('llm_apis.providers.openai_client.AsyncOpenAI') as mock_openai:
            # Mock response
            mock_response = MagicMock()
            mock_response.data = [MagicMock(), MagicMock()]
            mock_response.data[0].embedding = [0.1, 0.2, 0.3]
            mock_response.data[0].index = 0
            mock_response.data[1].embedding = [0.4, 0.5, 0.6]
            mock_response.data[1].index = 1
            mock_response.model = "text-embedding-ada-002"
            mock_response.usage = MagicMock()
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.total_tokens = 10
            
            # Mock client
            mock_client_instance = AsyncMock()
            mock_client_instance.embeddings.create.return_value = mock_response
            mock_openai.return_value = mock_client_instance
            
            # Test
            client = OpenAIClient(api_key="test-key")
            response = await client.get_embeddings(
                texts=["Hello", "World"],
                model="text-embedding-ada-002"
            )
            
            assert len(response.data) == 2
            assert response.data[0].embedding == [0.1, 0.2, 0.3]
            assert response.data[1].embedding == [0.4, 0.5, 0.6]
            assert response.model == "text-embedding-ada-002"
            assert response.provider == "openai"
    
    def test_get_available_models(self):
        """Test getting available models."""
        client = OpenAIClient(api_key="test-key")
        models = client.get_available_models()
        
        assert isinstance(models, list)
        assert "gpt-4" in models
        assert "gpt-3.5-turbo" in models
        assert "text-embedding-ada-002" in models
    
    def test_error_handling(self):
        """Test error handling and conversion."""
        client = OpenAIClient(api_key="test-key")
        
        # Test authentication error
        auth_error = Exception("401 authentication failed")
        with pytest.raises(AuthenticationError):
            client._handle_api_error(auth_error)
        
        # Test rate limit error
        rate_error = Exception("429 rate limit exceeded")
        with pytest.raises(RateLimitError):
            client._handle_api_error(rate_error)