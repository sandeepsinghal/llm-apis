# LLM APIs

A unified Python wrapper for various Large Language Model provider APIs including OpenAI, Anthropic, Perplexity, and Ollama.

## Features

- **Unified Interface**: Consistent API across different LLM providers
- **Async/Sync Support**: Both asynchronous and synchronous method calls
- **Streaming**: Support for streaming responses from all providers
- **Type Safety**: Full type hints and Pydantic model validation
- **Error Handling**: Comprehensive error handling and retry logic
- **Configuration**: Environment-based configuration management

## Supported Providers

- **OpenAI**: GPT-4, GPT-3.5-turbo, and embedding models
- **Anthropic**: Claude models (Opus, Sonnet, Haiku)
- **Perplexity**: Sonar and Llama models with web search
- **Ollama**: Local model deployment support

## Installation

```bash
pip install -e .
```

Or install with development dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Start

### Environment Setup

Create a `.env` file in your project root:

```env
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
PERPLEXITY_API_KEY=your_perplexity_api_key
OLLAMA_BASE_URL=http://localhost:11434
```

### Basic Usage

```python
import asyncio
from llm_apis import OpenAIClient, AnthropicClient

async def main():
    # OpenAI client
    openai_client = OpenAIClient()
    response = await openai_client.chat_completion(
        messages=[{"role": "user", "content": "Hello, world!"}],
        model="gpt-3.5-turbo"
    )
    print(response.choices[0].message["content"])
    
    # Anthropic client
    anthropic_client = AnthropicClient()
    response = await anthropic_client.chat_completion(
        messages=[{"role": "user", "content": "Hello, Claude!"}],
        model="claude-3-sonnet-20240229"
    )
    print(response.choices[0].message["content"])

asyncio.run(main())
```

### Streaming Responses

```python
import asyncio
from llm_apis import OpenAIClient

async def streaming_example():
    client = OpenAIClient()
    
    async for chunk in client.chat_completion_stream(
        messages=[{"role": "user", "content": "Write a short poem"}],
        model="gpt-3.5-turbo"
    ):
        content = chunk.choices[0].message["content"]
        if content:
            print(content, end="", flush=True)

asyncio.run(streaming_example())
```

### Synchronous Usage

```python
from llm_apis import OpenAIClient

client = OpenAIClient()
response = client.chat_completion_sync(
    messages=[{"role": "user", "content": "Hello!"}],
    model="gpt-3.5-turbo"
)
print(response.choices[0].message["content"])
```

## API Reference

### Base Client

All provider clients inherit from `BaseLLMClient` and implement:

- `chat_completion(messages, model, **kwargs)` - Generate chat completion
- `chat_completion_stream(messages, model, **kwargs)` - Streaming chat completion
- `get_embeddings(texts, model, **kwargs)` - Get text embeddings (where supported)
- `get_available_models()` - List available models

### Provider-Specific Clients

#### OpenAIClient

```python
from llm_apis import OpenAIClient

client = OpenAIClient(api_key="optional_override")

# Chat completion
response = await client.chat_completion(
    messages=[{"role": "user", "content": "Hello"}],
    model="gpt-4",
    temperature=0.7,
    max_tokens=100
)

# Embeddings
embeddings = await client.get_embeddings(
    texts=["Hello world", "How are you?"],
    model="text-embedding-ada-002"
)
```

#### AnthropicClient

```python
from llm_apis import AnthropicClient

client = AnthropicClient(api_key="optional_override")

response = await client.chat_completion(
    messages=[
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"}
    ],
    model="claude-3-sonnet-20240229",
    max_tokens=1024
)
```

#### PerplexityClient

```python
from llm_apis import PerplexityClient

client = PerplexityClient(api_key="your_api_key")

response = await client.chat_completion(
    messages=[{"role": "user", "content": "What's the weather like?"}],
    model="llama-3.1-sonar-small-128k-online"
)
```

#### OllamaClient

```python
from llm_apis import OllamaClient

client = OllamaClient(base_url="http://localhost:11434")

response = await client.chat_completion(
    messages=[{"role": "user", "content": "Hello"}],
    model="llama2"
)

# Get locally available models
models = await client.get_available_models()
```

## Configuration

Configuration can be managed through environment variables or programmatically:

```python
from llm_apis.utils import config

# Update provider configuration
config.update_config('openai', {
    'timeout': 60,
    'max_retries': 5
})

# Get provider configuration
openai_config = config.get_provider_config('openai')
```

## Error Handling

The package provides comprehensive error handling:

```python
from llm_apis import OpenAIClient
from llm_apis.exceptions import (
    AuthenticationError,
    RateLimitError,
    InvalidRequestError,
    ModelNotFoundError
)

client = OpenAIClient()

try:
    response = await client.chat_completion(
        messages=[{"role": "user", "content": "Hello"}],
        model="gpt-4"
    )
except AuthenticationError:
    print("Invalid API key")
except RateLimitError as e:
    print(f"Rate limited. Retry after: {e.retry_after}")
except ModelNotFoundError:
    print("Model not available")
except InvalidRequestError:
    print("Invalid request parameters")
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black llm_apis/
isort llm_apis/
```

### Type Checking

```bash
mypy llm_apis/
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## Support

For issues and questions, please open an issue on GitHub.