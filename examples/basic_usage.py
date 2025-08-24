"""
Basic usage examples for LLM APIs.
"""

import asyncio
from llm_apis import OpenAIClient, AnthropicClient, PerplexityClient, OllamaClient


async def openai_example():
    """Example using OpenAI client."""
    print("=== OpenAI Example ===")
    
    try:
        client = OpenAIClient()
        
        # Chat completion
        response = await client.chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"}
            ],
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=100
        )
        
        print(f"Response: {response.choices[0].message['content']}")
        print(f"Model: {response.model}")
        if response.usage:
            print(f"Tokens used: {response.usage.total_tokens}")
        
    except Exception as e:
        print(f"OpenAI Error: {e}")


async def anthropic_example():
    """Example using Anthropic client."""
    print("\n=== Anthropic Example ===")
    
    try:
        client = AnthropicClient()
        
        response = await client.chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain quantum computing in simple terms."}
            ],
            model="claude-3-sonnet-20240229",
            max_tokens=150
        )
        
        print(f"Response: {response.choices[0].message['content']}")
        print(f"Model: {response.model}")
        
    except Exception as e:
        print(f"Anthropic Error: {e}")


async def perplexity_example():
    """Example using Perplexity client."""
    print("\n=== Perplexity Example ===")
    
    try:
        client = PerplexityClient()
        
        response = await client.chat_completion(
            messages=[
                {"role": "user", "content": "What are the latest developments in AI research?"}
            ],
            model="llama-3.1-sonar-small-128k-online",
            temperature=0.7,
            max_tokens=200
        )
        
        print(f"Response: {response.choices[0].message['content']}")
        print(f"Model: {response.model}")
        
    except Exception as e:
        print(f"Perplexity Error: {e}")


async def ollama_example():
    """Example using Ollama client."""
    print("\n=== Ollama Example ===")
    
    try:
        client = OllamaClient()
        
        # Get available models
        models = await client.get_available_models()
        print(f"Available models: {models}")
        
        if models:
            # Use the first available model
            model = models[0]
            response = await client.chat_completion(
                messages=[
                    {"role": "user", "content": "Tell me a joke"}
                ],
                model=model,
                temperature=0.8
            )
            
            print(f"Response: {response.choices[0].message['content']}")
            print(f"Model: {response.model}")
        
    except Exception as e:
        print(f"Ollama Error: {e}")


async def embeddings_example():
    """Example of getting embeddings."""
    print("\n=== Embeddings Example ===")
    
    try:
        client = OpenAIClient()
        
        texts = [
            "The weather is beautiful today",
            "I love programming with Python",
            "Machine learning is fascinating"
        ]
        
        embeddings_response = await client.get_embeddings(
            texts=texts,
            model="text-embedding-ada-002"
        )
        
        print(f"Generated {len(embeddings_response.data)} embeddings")
        for i, embedding_data in enumerate(embeddings_response.data):
            print(f"Text {i+1}: {len(embedding_data.embedding)} dimensions")
        
    except Exception as e:
        print(f"Embeddings Error: {e}")


async def main():
    """Run all examples."""
    await openai_example()
    await anthropic_example()
    await perplexity_example()
    await ollama_example()
    await embeddings_example()


if __name__ == "__main__":
    asyncio.run(main())