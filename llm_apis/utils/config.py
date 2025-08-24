"""
Configuration management for LLM APIs.
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration manager for LLM API clients."""
    
    def __init__(self):
        """Initialize configuration with environment variables."""
        self._config = {}
        self._load_env_config()
    
    def _load_env_config(self):
        """Load configuration from environment variables."""
        # OpenAI configuration
        self._config['openai'] = {
            'api_key': os.getenv('OPENAI_API_KEY'),
            'base_url': os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
            'timeout': int(os.getenv('OPENAI_TIMEOUT', '30')),
            'max_retries': int(os.getenv('OPENAI_MAX_RETRIES', '3'))
        }
        
        # Anthropic configuration
        self._config['anthropic'] = {
            'api_key': os.getenv('ANTHROPIC_API_KEY'),
            'base_url': os.getenv('ANTHROPIC_BASE_URL', 'https://api.anthropic.com'),
            'timeout': int(os.getenv('ANTHROPIC_TIMEOUT', '30')),
            'max_retries': int(os.getenv('ANTHROPIC_MAX_RETRIES', '3'))
        }
        
        # Perplexity configuration
        self._config['perplexity'] = {
            'api_key': os.getenv('PERPLEXITY_API_KEY'),
            'base_url': os.getenv('PERPLEXITY_BASE_URL', 'https://api.perplexity.ai'),
            'timeout': int(os.getenv('PERPLEXITY_TIMEOUT', '30')),
            'max_retries': int(os.getenv('PERPLEXITY_MAX_RETRIES', '3'))
        }
        
        # Ollama configuration
        self._config['ollama'] = {
            'base_url': os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
            'timeout': int(os.getenv('OLLAMA_TIMEOUT', '30')),
            'max_retries': int(os.getenv('OLLAMA_MAX_RETRIES', '3'))
        }
        
        # Global configuration
        self._config['global'] = {
            'default_timeout': int(os.getenv('LLM_DEFAULT_TIMEOUT', '30')),
            'default_max_retries': int(os.getenv('LLM_DEFAULT_MAX_RETRIES', '3')),
            'enable_logging': os.getenv('LLM_ENABLE_LOGGING', 'false').lower() == 'true',
            'log_level': os.getenv('LLM_LOG_LEVEL', 'INFO').upper()
        }
    
    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """
        Get configuration for a specific provider.
        
        Args:
            provider: Provider name (openai, anthropic, perplexity, ollama)
            
        Returns:
            Configuration dictionary for the provider
        """
        return self._config.get(provider, {})
    
    def get_global_config(self) -> Dict[str, Any]:
        """
        Get global configuration.
        
        Returns:
            Global configuration dictionary
        """
        return self._config.get('global', {})
    
    def update_config(self, provider: str, config: Dict[str, Any]):
        """
        Update configuration for a provider.
        
        Args:
            provider: Provider name
            config: Configuration updates
        """
        if provider not in self._config:
            self._config[provider] = {}
        self._config[provider].update(config)
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Get API key for a provider.
        
        Args:
            provider: Provider name
            
        Returns:
            API key if available
        """
        provider_config = self.get_provider_config(provider)
        return provider_config.get('api_key')


# Global configuration instance
config = Config()