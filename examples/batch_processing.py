"""
Batch processing examples for LLM APIs.
"""

import asyncio
import time
from typing import List, Dict, Any
from llm_apis import OpenAIClient, AnthropicClient


async def batch_chat_completions():
    """Example of processing multiple chat completions in batches."""
    print("=== Batch Chat Completions ===")
    
    # Sample prompts for batch processing
    prompts = [
        "What is the capital of France?",
        "Explain photosynthesis in one sentence.",
        "What is 15 + 27?",
        "Name three programming languages.",
        "What color is the sky?",
        "How many days are in a week?",
        "What is the largest planet in our solar system?",
        "What does CPU stand for?",
        "Name a popular web browser.",
        "What is the freezing point of water in Celsius?"
    ]
    
    try:
        client = OpenAIClient()
        
        print(f"Processing {len(prompts)} prompts...")
        start_time = time.time()
        
        # Process all prompts concurrently
        tasks = []
        for i, prompt in enumerate(prompts):
            task = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=50
            )
            tasks.append((i, prompt, task))
        
        # Wait for all completions
        results = []
        for i, prompt, task in tasks:
            try:
                response = await task
                content = response.choices[0].message["content"]
                results.append((i, prompt, content, response.usage))
            except Exception as e:
                results.append((i, prompt, f"Error: {e}", None))
        
        end_time = time.time()
        
        # Display results
        print(f"\nCompleted in {end_time - start_time:.2f} seconds")
        print("-" * 80)
        
        total_tokens = 0
        for i, prompt, response, usage in results:
            print(f"Prompt {i+1}: {prompt}")
            print(f"Response: {response}")
            if usage:
                print(f"Tokens: {usage.total_tokens}")
                total_tokens += usage.total_tokens
            print("-" * 40)
        
        print(f"Total tokens used: {total_tokens}")
        
    except Exception as e:
        print(f"Batch processing error: {e}")


async def batch_embeddings():
    """Example of processing multiple texts for embeddings."""
    print("\n=== Batch Embeddings ===")
    
    texts = [
        "The weather is sunny today",
        "I love programming in Python",
        "Machine learning is fascinating",
        "The ocean is deep and mysterious",
        "Books are a source of knowledge",
        "Music brings joy to life",
        "Travel broadens the mind",
        "Exercise is good for health",
        "Art expresses human creativity",
        "Science explains the world"
    ]
    
    try:
        client = OpenAIClient()
        
        print(f"Generating embeddings for {len(texts)} texts...")
        start_time = time.time()
        
        # Process embeddings in batch
        response = await client.get_embeddings(
            texts=texts,
            model="text-embedding-ada-002"
        )
        
        end_time = time.time()
        
        print(f"Completed in {end_time - start_time:.2f} seconds")
        print(f"Generated {len(response.data)} embeddings")
        print(f"Embedding dimensions: {len(response.data[0].embedding)}")
        
        if response.usage:
            print(f"Tokens used: {response.usage.total_tokens}")
        
        # Show similarity example
        print("\nCalculating similarities...")
        embedding1 = response.data[0].embedding  # "The weather is sunny today"
        embedding2 = response.data[1].embedding  # "I love programming in Python"
        embedding3 = response.data[3].embedding  # "The ocean is deep and mysterious"
        
        def cosine_similarity(a: List[float], b: List[float]) -> float:
            """Calculate cosine similarity between two vectors."""
            dot_product = sum(x * y for x, y in zip(a, b))
            magnitude_a = sum(x * x for x in a) ** 0.5
            magnitude_b = sum(x * x for x in b) ** 0.5
            return dot_product / (magnitude_a * magnitude_b)
        
        sim_1_2 = cosine_similarity(embedding1, embedding2)
        sim_1_3 = cosine_similarity(embedding1, embedding3)
        
        print(f"Similarity between '{texts[0]}' and '{texts[1]}': {sim_1_2:.4f}")
        print(f"Similarity between '{texts[0]}' and '{texts[3]}': {sim_1_3:.4f}")
        
    except Exception as e:
        print(f"Batch embeddings error: {e}")


async def process_with_different_providers():
    """Example of processing the same prompts with different providers."""
    print("\n=== Multi-Provider Processing ===")
    
    prompts = [
        "What is artificial intelligence?",
        "Explain quantum computing briefly.",
        "What are the benefits of renewable energy?"
    ]
    
    providers = [
        ("OpenAI", OpenAIClient(), "gpt-3.5-turbo"),
        ("Anthropic", AnthropicClient(), "claude-3-sonnet-20240229")
    ]
    
    for provider_name, client, model in providers:
        try:
            print(f"\n--- {provider_name} Results ---")
            
            tasks = []
            for prompt in prompts:
                task = client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    temperature=0.7,
                    max_tokens=100
                )
                tasks.append((prompt, task))
            
            for prompt, task in tasks:
                try:
                    response = await task
                    content = response.choices[0].message["content"]
                    print(f"Q: {prompt}")
                    print(f"A: {content}")
                    print("-" * 40)
                except Exception as e:
                    print(f"Error for '{prompt}': {e}")
                    print("-" * 40)
        
        except Exception as e:
            print(f"{provider_name} provider error: {e}")


async def rate_limited_batch_processing():
    """Example of batch processing with rate limiting."""
    print("\n=== Rate-Limited Batch Processing ===")
    
    # Generate more prompts for rate limiting example
    prompts = [f"Tell me an interesting fact about the number {i}." for i in range(1, 21)]
    
    try:
        client = OpenAIClient()
        
        print(f"Processing {len(prompts)} prompts with rate limiting...")
        
        # Process in smaller batches with delays
        batch_size = 5
        delay_between_batches = 1  # seconds
        
        all_results = []
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            print(f"Processing batch {i // batch_size + 1}...")
            
            # Process current batch
            tasks = []
            for prompt in batch:
                task = client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    model="gpt-3.5-turbo",
                    temperature=0.7,
                    max_tokens=50
                )
                tasks.append((prompt, task))
            
            # Wait for batch completion
            batch_results = []
            for prompt, task in tasks:
                try:
                    response = await task
                    content = response.choices[0].message["content"]
                    batch_results.append((prompt, content))
                except Exception as e:
                    batch_results.append((prompt, f"Error: {e}"))
            
            all_results.extend(batch_results)
            
            # Delay before next batch (except for the last batch)
            if i + batch_size < len(prompts):
                await asyncio.sleep(delay_between_batches)
        
        # Display results
        print(f"\nProcessed {len(all_results)} prompts:")
        for i, (prompt, response) in enumerate(all_results, 1):
            print(f"{i}. {prompt}")
            print(f"   {response}")
            print()
        
    except Exception as e:
        print(f"Rate-limited processing error: {e}")


async def main():
    """Run all batch processing examples."""
    await batch_chat_completions()
    await batch_embeddings()
    await process_with_different_providers()
    await rate_limited_batch_processing()


if __name__ == "__main__":
    asyncio.run(main())