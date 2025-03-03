import os
import requests
import json
from typing import Dict, Any, List, Optional

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

def get_anthropic_api_key() -> str:
    """Retrieve the Anthropic API key from environment variables."""
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    return api_key

def ask_claude(
    prompt: str,
    model: str = "claude-3-opus-20240229",
    max_tokens: int = 1000,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Send a prompt to Anthropic's Claude model and return the response.
    
    Args:
        prompt: The user message to send to Claude
        model: The Claude model to use (default: claude-3-opus-20240229)
        max_tokens: Maximum number of tokens in the response
        temperature: Controls randomness (0.0-1.0)
        system_prompt: Optional system instructions
    
    Returns:
        Dict containing the full API response
    """
    headers = {
        "x-api-key": get_anthropic_api_key(),
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    if system_prompt:
        payload["system"] = system_prompt
    
    response = requests.post(ANTHROPIC_API_URL, headers=headers, json=payload)
    
    if response.status_code != 200:
        raise Exception(f"API request failed with status {response.status_code}: {response.text}")
    
    return response.json()

def get_response_text(response: Dict[str, Any]) -> str:
    """
    Extract the text content from a Claude API response.
    
    Args:
        response: The API response from Claude
        
    Returns:
        The text content of the response
    """
    if "content" not in response:
        raise ValueError("Invalid response format")
    
    for block in response.get("content", []):
        if block.get("type") == "text":
            return block.get("text", "")
    
    return ""

def test_anthropic_api():
    """Test the Anthropic API by asking a simple question."""
    try:
        question = "What are the three laws of robotics by Isaac Asimov?"
        print(f"Question: {question}")
        print("Sending request to Anthropic API...")
        
        response = ask_claude(
            prompt=question,
            model="claude-3-haiku-20240307",  # Using a smaller model for testing
            max_tokens=300,
            temperature=0.7
        )
        
        print("\nResponse from Claude:")
        print("---------------------")
        content = response.get("content", [])
        for block in content:
            if block.get("type") == "text":
                print(block.get("text"))
        
        print("\nFull API response:")
        print(json.dumps(response, indent=2))
        
        return True
    except Exception as e:
        print(f"Error testing Anthropic API: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing Anthropic API integration...")
    test_anthropic_api()
