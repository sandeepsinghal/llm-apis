import requests
import os

class PerplexityClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('PERPLEXITY_API_KEY')
        self.api_url = 'https://api.perplexity.ai/chat/completions'

    def send_prompt(self, prompt, model='sonar-pro'):
        response = requests.post(
            self.api_url,
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            },
            json={
                'model': model,
                'messages': [
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]
            }
        )
        return response.json()

if __name__ == '__main__':
    # Test case
    client = PerplexityClient()
    result = client.send_prompt("What are the major AI developments and announcements from today across the tech industry?")
    print(result)