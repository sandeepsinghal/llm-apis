"""
Tests for Anthropic client.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from llm_apis.providers.anthropic_client import AnthropicClient
from llm_apis.exceptions import AuthenticationError


class TestAnthropicClient:
    """Test cases for Anthropic client."""
    
    def test_init_without_api_key(self):
        """Test initialization without API key raises error."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(AuthenticationError):
                AnthropicClient()
    
    def test_init_with_api_key(self):
        """Test initialization with API key."""
        client = AnthropicClient(api_key="test-key")
        assert client.api_key == "test-key"
        assert client.provider == "anthropic"
    
    @pytest.mark.asyncio
    async def test_chat_completion_success(self):
        """Test successful chat completion."""
        with patch('llm_apis.providers.anthropic_client.AsyncAnthropic') as mock_anthropic:
            # Mock response
            mock_response = MagicMock()
            mock_response.id = "test-id"
            mock_response.content = [MagicMock()]
            mock_response.content[0].text = "Hello, human!"
            mock_response.stop_reason = "end_turn"
            mock_response.usage = MagicMock()
            mock_response.usage.input_tokens = 10
            mock_response.usage.output_tokens = 5
            
            # Mock client
            mock_client_instance = AsyncMock()
            mock_client_instance.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client_instance
            
            # Test
            client = AnthropicClient(api_key="test-key")
            response = await client.chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                model="claude-3-sonnet-20240229"
            )
            
            assert response.id == "test-id"
            assert response.model == "claude-3-sonnet-20240229"
            assert response.choices[0].message["content"] == "Hello, human!"
            assert response.usage.prompt_tokens == 10
            assert response.usage.completion_tokens == 5
            assert response.provider == "anthropic"
    
    @pytest.mark.asyncio
    async def test_embeddings_not_implemented(self):
        """Test that embeddings raise NotImplementedError."""
        client = AnthropicClient(api_key="test-key")
        
        with pytest.raises(NotImplementedError):
            await client.get_embeddings(
                texts=["Hello", "World"],
                model="claude-3-sonnet-20240229"
            )
    
    def test_get_available_models(self):
        """Test getting available models."""
        client = AnthropicClient(api_key="test-key")
        models = client.get_available_models()
        
        assert isinstance(models, list)
        assert "claude-3-opus-20240229" in models
        assert "claude-3-sonnet-20240229" in models
        assert "claude-3-haiku-20240307" in models