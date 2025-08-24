"""
Streaming response examples for LLM APIs.
"""

import asyncio
from llm_apis import OpenAIClient, AnthropicClient, PerplexityClient, OllamaClient


async def openai_streaming_example():
    """Example of streaming responses from OpenAI."""
    print("=== OpenAI Streaming Example ===")
    
    try:
        client = OpenAIClient()
        
        print("User: Write a short story about a robot learning to paint.")
        print("Assistant: ", end="", flush=True)
        
        async for chunk in client.chat_completion_stream(
            messages=[
                {"role": "user", "content": "Write a short story about a robot learning to paint."}
            ],
            model="gpt-3.5-turbo",
            temperature=0.8,
            max_tokens=300
        ):
            content = chunk.choices[0].message.get("content", "")
            if content:
                print(content, end="", flush=True)
        
        print("\n")
        
    except Exception as e:
        print(f"OpenAI Streaming Error: {e}")


async def anthropic_streaming_example():
    """Example of streaming responses from Anthropic."""
    print("\n=== Anthropic Streaming Example ===")
    
    try:
        client = AnthropicClient()
        
        print("User: Explain the process of photosynthesis step by step.")
        print("Assistant: ", end="", flush=True)
        
        async for chunk in client.chat_completion_stream(
            messages=[
                {"role": "user", "content": "Explain the process of photosynthesis step by step."}
            ],
            model="claude-3-sonnet-20240229",
            max_tokens=400
        ):
            content = chunk.choices[0].message.get("content", "")
            if content:
                print(content, end="", flush=True)
        
        print("\n")
        
    except Exception as e:
        print(f"Anthropic Streaming Error: {e}")


async def perplexity_streaming_example():
    """Example of streaming responses from Perplexity."""
    print("\n=== Perplexity Streaming Example ===")
    
    try:
        client = PerplexityClient()
        
        print("User: What are the current trends in renewable energy?")
        print("Assistant: ", end="", flush=True)
        
        async for chunk in client.chat_completion_stream(
            messages=[
                {"role": "user", "content": "What are the current trends in renewable energy?"}
            ],
            model="llama-3.1-sonar-small-128k-online",
            temperature=0.7,
            max_tokens=300
        ):
            content = chunk.choices[0].message.get("content", "")
            if content:
                print(content, end="", flush=True)
        
        print("\n")
        
    except Exception as e:
        print(f"Perplexity Streaming Error: {e}")


async def ollama_streaming_example():
    """Example of streaming responses from Ollama."""
    print("\n=== Ollama Streaming Example ===")
    
    try:
        client = OllamaClient()
        
        # Get available models
        models = await client.get_available_models()
        
        if models:
            model = models[0]
            print(f"User: Tell me about the history of computers (using {model}).")
            print("Assistant: ", end="", flush=True)
            
            async for chunk in client.chat_completion_stream(
                messages=[
                    {"role": "user", "content": "Tell me about the history of computers."}
                ],
                model=model,
                temperature=0.7
            ):
                content = chunk.choices[0].message.get("content", "")
                if content:
                    print(content, end="", flush=True)
            
            print("\n")
        else:
            print("No Ollama models available. Make sure Ollama is running.")
        
    except Exception as e:
        print(f"Ollama Streaming Error: {e}")


async def compare_providers_streaming():
    """Compare streaming responses from different providers."""
    print("\n=== Provider Comparison ===")
    
    question = "What is the meaning of life?"
    
    providers = [
        ("OpenAI", OpenAIClient(), "gpt-3.5-turbo"),
        ("Anthropic", AnthropicClient(), "claude-3-sonnet-20240229"),
        ("Perplexity", PerplexityClient(), "llama-3.1-sonar-small-128k-online")
    ]
    
    for provider_name, client, model in providers:
        try:
            print(f"\n--- {provider_name} Response ---")
            print("Assistant: ", end="", flush=True)
            
            async for chunk in client.chat_completion_stream(
                messages=[{"role": "user", "content": question}],
                model=model,
                temperature=0.7,
                max_tokens=150
            ):
                content = chunk.choices[0].message.get("content", "")
                if content:
                    print(content, end="", flush=True)
            
            print("\n")
            
        except Exception as e:
            print(f"{provider_name} Error: {e}")


async def interactive_chat():
    """Interactive chat example with streaming."""
    print("\n=== Interactive Chat (OpenAI) ===")
    print("Type 'quit' to exit")
    
    try:
        client = OpenAIClient()
        conversation = []
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                break
            
            if not user_input:
                continue
            
            conversation.append({"role": "user", "content": user_input})
            
            print("Assistant: ", end="", flush=True)
            
            assistant_response = ""
            async for chunk in client.chat_completion_stream(
                messages=conversation,
                model="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=200
            ):
                content = chunk.choices[0].message.get("content", "")
                if content:
                    assistant_response += content
                    print(content, end="", flush=True)
            
            print("")  # New line after response
            
            if assistant_response:
                conversation.append({"role": "assistant", "content": assistant_response})
        
    except Exception as e:
        print(f"Interactive Chat Error: {e}")


async def main():
    """Run all streaming examples."""
    await openai_streaming_example()
    await anthropic_streaming_example()
    await perplexity_streaming_example()
    await ollama_streaming_example()
    await compare_providers_streaming()
    
    # Uncomment to run interactive chat
    # await interactive_chat()


if __name__ == "__main__":
    asyncio.run(main())